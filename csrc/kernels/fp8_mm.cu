// Fused BF16 -> E4M3 MMA -> BF16 matrix multiplication

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace {

constexpr int kMmaK = 32;
constexpr int kWarps = 8;
// Fast forward path: 128x64 CTA, 64x16 warp tile, 2-stage pipeline, dynamic
// shared memory. Mirrors the CUTLASS 58_ada_fp8_gemm threadblock geometry
// while keeping the fused BF16->FP8 quantize path. The FP8 tile overwrites
// the BF16 staging area in place. L20 opts in to only 101376 B shared per
// block; K=32 keeps the footprint at 24576 B so four CTAs/SM stay resident.
constexpr int kFastBlockM = 128;
constexpr int kFastBlockN = 64;
constexpr int kFastK = 32;
constexpr int kFastStages = 2;
constexpr int kFastSmemBytes =
    kFastStages * (kFastBlockM * kFastK * 2 + kFastBlockN * kFastK * 2);

__device__ __forceinline__ unsigned pack_fp8x4_vector(float x0, float x1,
                                                       float x2, float x3) {
    const auto low = __nv_cvt_float2_to_fp8x2(
        make_float2(x0, x1), __NV_SATFINITE, __NV_E4M3);
    const auto high = __nv_cvt_float2_to_fp8x2(
        make_float2(x2, x3), __NV_SATFINITE, __NV_E4M3);
    return static_cast<unsigned>(low) | (static_cast<unsigned>(high) << 16);
}


__device__ __forceinline__ void mma_fp8_16832(float d[4],
                                               const unsigned a[4],
                                               const unsigned b[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void atomic_max_float(float* destination,
                                                  float value) {
    if (destination)
        atomicMax(reinterpret_cast<unsigned*>(destination), __float_as_uint(value));
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

// Block-wide max reduction of a per-warp tracked value, then an atomic
// update of the global amax slot when `track` is set.
template <int NWarps>
__device__ __forceinline__ void block_reduce_amax(float& local, float* slots,
                                                  int warp, int lane,
                                                  bool track, float* global) {
    local = warp_reduce_max(local);
    if (lane == 0) slots[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float value = lane < NWarps ? slots[lane] : 0.0f;
        value = warp_reduce_max(value);
        if (lane == 0 && track && global) atomic_max_float(global, value);
    }
}

// One thread moves eight BF16 values (16 bytes). The async copy is issued
// through a uint4-shaped pointer so the source and destination are both
// naturally 128-bit aligned for contiguous forward GEMMs.
__device__ __forceinline__ void cp_async_bf16_8(
    __nv_bfloat16* destination, const __nv_bfloat16* source, bool valid) {
    const unsigned shared_address = __cvta_generic_to_shared(destination);
    const uint4* source_vec = reinterpret_cast<const uint4*>(source);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_address), "l"(source_vec),
                    "r"(valid ? 16 : 0));
}

template <bool TrackAmax = true>
__device__ __forceinline__ unsigned load_fp8x4_from_bf16(
    const __nv_bfloat16* source, float scale_inv, float& amax,
    bool track_amax = true) {
    float x0 = __bfloat162float(source[0]);
    float x1 = __bfloat162float(source[1]);
    float x2 = __bfloat162float(source[2]);
    float x3 = __bfloat162float(source[3]);
    if constexpr (TrackAmax) {
        if (track_amax) {
        amax = fmaxf(amax, fmaxf(fabsf(x0), fmaxf(fabsf(x1),
                             fmaxf(fabsf(x2), fabsf(x3)))));
        }
    }
    return pack_fp8x4_vector(x0 * scale_inv, x1 * scale_inv,
                             x2 * scale_inv, x3 * scale_inv);
}


template <bool AddBias, bool TrackAmax>
__global__ void fused_fp8_gemm_fast_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ bias,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    float* __restrict__ amax_a,
    float* __restrict__ amax_b,
    int64_t m, int64_t n, int64_t k) {
    extern __shared__ char smem[];
    constexpr int a_stride = kFastBlockM * kFastK;
    constexpr int b_stride = kFastBlockN * kFastK;
    constexpr int b_bf16_offset = kFastStages * a_stride;
    auto* a_bf16 = reinterpret_cast<__nv_bfloat16*>(smem);
    auto* b_bf16 =
        reinterpret_cast<__nv_bfloat16*>(smem + b_bf16_offset * 2);
    __shared__ float warp_amax_a[kWarps];
    __shared__ float warp_amax_b[kWarps];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    constexpr int warps_n = kFastBlockN / 16;
    const int warp_m = warp / warps_n;
    const int warp_n = warp % warps_n;
    const int64_t row_base =
        blockIdx.y * kFastBlockM + warp_m * 64 + group;
    const int64_t output_col =
        blockIdx.x * kFastBlockN + warp_n * 16 + thread_in_group * 2;
    const float sa = *scale_a;
    const float sb = *scale_b;
    const float inv_a = 1.0f / sa;
    const float inv_b = 1.0f / sb;
    float local_amax_a = 0.0f;
    float local_amax_b = 0.0f;
    float acc[4 * 4 * 2] = {};

    const bool track_amax_a = TrackAmax && blockIdx.x == 0;
    const bool track_amax_b = TrackAmax && blockIdx.y == 0;
    // Each thread issues 8 A chunks and 4 B chunks of 8 BF16 (16B) per stage.
    auto load_tile = [&](int stage, int64_t k_base) {
        const int r0 = tid >> 2;
        const int c0 = (tid & 3) * 8;
#pragma unroll
        for (int j = 0; j < kFastK / 32; ++j) {
            const int col = c0 + 32 * j;
            const bool full_chunk = k_base + col + 7 < k;
            const int64_t a_row = blockIdx.y * kFastBlockM + r0;
            const int64_t b_row = blockIdx.x * kFastBlockN + r0;
            auto* a_dst = &a_bf16[stage * a_stride + r0 * kFastK + col];
            auto* b_dst = &b_bf16[stage * b_stride + r0 * kFastK + col];
            const auto* a_ptr = a + a_row * k + k_base + col;
            const auto* b_ptr = b + b_row * k + k_base + col;
            const bool full_a = a_row < m && full_chunk;
            const bool full_b = b_row < n && full_chunk;
            const bool aligned_a =
                (reinterpret_cast<uintptr_t>(a_ptr) & 15) == 0;
            const bool aligned_b =
                (reinterpret_cast<uintptr_t>(b_ptr) & 15) == 0;
            if (full_a && aligned_a) {
                cp_async_bf16_8(a_dst, a_ptr, true);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    a_dst[i] = a_row < m && k_base + col + i < k
                                   ? a_ptr[i]
                                   : __float2bfloat16(0.0f);
                }
            }
            if (full_b && aligned_b) {
                cp_async_bf16_8(b_dst, b_ptr, true);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    b_dst[i] = b_row < n && k_base + col + i < k
                                   ? b_ptr[i]
                                   : __float2bfloat16(0.0f);
                }
            }
            if (r0 + 64 < kFastBlockM) {
                const int64_t a_row_hi = blockIdx.y * kFastBlockM + r0 + 64;
                auto* a_dst_hi =
                    &a_bf16[stage * a_stride + (r0 + 64) * kFastK + col];
                const auto* a_ptr_hi = a + a_row_hi * k + k_base + col;
                const bool full_a_hi = a_row_hi < m && full_chunk;
                const bool aligned_a_hi =
                    (reinterpret_cast<uintptr_t>(a_ptr_hi) & 15) == 0;
                if (full_a_hi && aligned_a_hi) {
                    cp_async_bf16_8(a_dst_hi, a_ptr_hi, true);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        a_dst_hi[i] = a_row_hi < m && k_base + col + i < k
                                          ? a_ptr_hi[i]
                                          : __float2bfloat16(0.0f);
                    }
                }
            }
        }
    };

    // Quantize must place each 4-BP8 group at the byte offset the MMA
    // fragment reads: 8*(lane&3) + 64*k_seg for a row. With in-place storage
    // (fp8 element k lives at byte 2k), the BF16 column of a group is
    // 4*(tid&7) + 32*j, so partition by 4-element groups instead of the
    // 8-element cp.async chunks.
    auto quantize_tile = [&](int stage) {
        const int r0 = tid >> 3;
        const int c0 = (tid & 7) * 4;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int row = r0 + 32 * s;
            auto* a_src = &a_bf16[stage * a_stride + row * kFastK + c0];
            auto* a_dst = reinterpret_cast<unsigned*>(a_src);
#pragma unroll
            for (int j = 0; j < kFastK / 32; ++j) {
                a_dst[16 * j] = load_fp8x4_from_bf16<TrackAmax>(
                    a_src + 32 * j, inv_a, local_amax_a, track_amax_a);
            }
        }
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int row = r0 + 32 * s;
            auto* b_src = &b_bf16[stage * b_stride + row * kFastK + c0];
            auto* b_dst = reinterpret_cast<unsigned*>(b_src);
#pragma unroll
            for (int j = 0; j < kFastK / 32; ++j) {
                b_dst[16 * j] = load_fp8x4_from_bf16<TrackAmax>(
                    b_src + 32 * j, inv_b, local_amax_b, track_amax_b);
            }
        }
    };

    const int64_t tile_count = (k + kFastK - 1) / kFastK;
    load_tile(0, 0);
    asm volatile("cp.async.commit_group;");
    if (tile_count > 1) {
        load_tile(1, kFastK);
        asm volatile("cp.async.commit_group;");
    }
    for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const int stage = static_cast<int>(tile_index % kFastStages);
        if (tile_index + 1 == tile_count) {
            asm volatile("cp.async.wait_group 0;");
        } else {
            asm volatile("cp.async.wait_group 1;");
        }
        // wait_group only waits for this thread's async copies. All threads
        // must finish loading before the tile is read by the CTA.
        __syncthreads();
        quantize_tile(stage);
        __syncthreads();

        // Four m16n8k32 MMA segments per 128-K stage.
#pragma unroll
        for (int k_seg = 0; k_seg < kFastK / kMmaK; ++k_seg) {
            const int frag_col = thread_in_group * 4 + k_seg * 32;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int b_row = warp_n * 16 + nt * 8 + group;
                unsigned b_frag[2];
                b_frag[0] = *reinterpret_cast<const unsigned*>(
                    &b_bf16[stage * b_stride + b_row * kFastK + frag_col]);
                b_frag[1] = *reinterpret_cast<const unsigned*>(
                    &b_bf16[stage * b_stride + b_row * kFastK + frag_col + 16]);
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const int a_row0 = warp_m * 64 + mt * 16 + group;
                    unsigned a_frag[4];
                    a_frag[0] = *reinterpret_cast<const unsigned*>(
                        &a_bf16[stage * a_stride + a_row0 * kFastK + frag_col]);
                    a_frag[1] = *reinterpret_cast<const unsigned*>(
                        &a_bf16[stage * a_stride + (a_row0 + 8) * kFastK + frag_col]);
                    a_frag[2] = *reinterpret_cast<const unsigned*>(
                        &a_bf16[stage * a_stride + a_row0 * kFastK + frag_col + 16]);
                    a_frag[3] = *reinterpret_cast<const unsigned*>(
                        &a_bf16[stage * a_stride + (a_row0 + 8) * kFastK + frag_col + 16]);
                    mma_fp8_16832(acc + (nt * 4 + mt) * 4, a_frag, b_frag);
                }
            }
        }
        __syncthreads();
        if (tile_index + 2 < tile_count) {
            load_tile(stage, (tile_index + 2) * kFastK);
            asm volatile("cp.async.commit_group;");
        }
    }

    if constexpr (TrackAmax) {
    block_reduce_amax<kWarps>(local_amax_a, warp_amax_a, warp, lane,
                              track_amax_a, amax_a);
    block_reduce_amax<kWarps>(local_amax_b, warp_amax_b, warp, lane,
                              track_amax_b, amax_b);
    }

    const float output_scale = sa * sb;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int64_t col = output_col + nt * 8;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int64_t row0 = row_base + mt * 16;
            const int64_t row1 = row0 + 8;
            float* tile_acc = acc + (nt * 4 + mt) * 4;
            if (col < n) {
                float bias0 = 0.0f;
                float bias1 = 0.0f;
                if constexpr (AddBias) {
                    bias0 = __bfloat162float(bias[col]);
                    if (col + 1 < n)
                        bias1 = __bfloat162float(bias[col + 1]);
                }
                if (row0 < m) {
                    out[row0 * n + col] =
                        __float2bfloat16(tile_acc[0] * output_scale + bias0);
                    if (col + 1 < n)
                        out[row0 * n + col + 1] = __float2bfloat16(
                            tile_acc[1] * output_scale + bias1);
                }
                if (row1 < m) {
                    out[row1 * n + col] =
                        __float2bfloat16(tile_acc[2] * output_scale + bias0);
                    if (col + 1 < n)
                        out[row1 * n + col + 1] = __float2bfloat16(
                            tile_acc[3] * output_scale + bias1);
                }
            }
        }
    }
}

// Pre-quantized FP8-in path: FP8 A/B read straight into shared memory (no
// BF16 staging, no inline quantization), FP32 accumulation, BF16 output.
// Same 128x64 CTA / 64x16 warp tile geometry as the fused kernel; the fp8
// tile is compact (row = kFastK bytes) so MMA fragments read directly.
constexpr int kPqBlockM = 128;
constexpr int kPqBlockN = 64;
constexpr int kPqK = 32;
constexpr int kPqStages = 3;

template <typename T>
__device__ __forceinline__ void cp_async_16b(T* destination,
                                             const T* source, bool valid) {
    const unsigned shared_address = __cvta_generic_to_shared(destination);
    const uint4* source_vec = reinterpret_cast<const uint4*>(source);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_address), "l"(source_vec),
                    "r"(valid ? 16 : 0));
}

template <bool OutFp8>
__global__ void fp8_mm_pq_kernel(
    const __nv_fp8_e4m3* __restrict__ a,
    const __nv_fp8_e4m3* __restrict__ b,
    __nv_bfloat16* __restrict__ out_bf16,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    const float scale, const float out_scale,
    int64_t m, int64_t n, int64_t k) {
    __shared__ __align__(16) __nv_fp8_e4m3 a_tile[kPqStages][kPqBlockM][kPqK];
    __shared__ __align__(16) __nv_fp8_e4m3 b_tile[kPqStages][kPqBlockN][kPqK];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    constexpr int warps_n = kPqBlockN / 16;
    const int warp_m = warp / warps_n;
    const int warp_n = warp % warps_n;
    const int64_t row_base = blockIdx.y * kPqBlockM + warp_m * 64 + group;
    const int64_t output_col =
        blockIdx.x * kPqBlockN + warp_n * 16 + thread_in_group * 2;
    float acc[4 * 4 * 2] = {};

    // One A chunk (16 FP8) per thread covers the 128x32 tile; the first 128
    // threads issue the 64x32 B chunks.
    auto load_tile = [&](int stage, int64_t k_base) {
        const int r0 = tid >> 1;
        const int c0 = (tid & 1) * 16;
        const bool full_chunk = k_base + c0 + 15 < k;
        const int64_t a_row = blockIdx.y * kPqBlockM + r0;
        auto* a_dst = &a_tile[stage][r0][c0];
        const auto* a_ptr = a + a_row * k + k_base + c0;
        const bool full_a = a_row < m && full_chunk;
        const bool aligned_a =
            (reinterpret_cast<uintptr_t>(a_ptr) & 15) == 0;
        if (full_a && aligned_a) {
            cp_async_16b(a_dst, a_ptr, true);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                a_dst[i] = a_row < m && k_base + c0 + i < k
                               ? a_ptr[i]
                               : __nv_fp8_e4m3(0.0f);
            }
        }
        if (tid < 128) {
            const int64_t b_row = blockIdx.x * kPqBlockN + r0;
            auto* b_dst = &b_tile[stage][r0][c0];
            const auto* b_ptr = b + b_row * k + k_base + c0;
            const bool full_b = b_row < n && full_chunk;
            const bool aligned_b =
                (reinterpret_cast<uintptr_t>(b_ptr) & 15) == 0;
            if (full_b && aligned_b) {
                cp_async_16b(b_dst, b_ptr, true);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    b_dst[i] = b_row < n && k_base + c0 + i < k
                                   ? b_ptr[i]
                                   : __nv_fp8_e4m3(0.0f);
                }
            }
        }
    };

    const int64_t tile_count = (k + kPqK - 1) / kPqK;
    load_tile(0, 0);
    asm volatile("cp.async.commit_group;");
    if (tile_count > 1) {
        load_tile(1, kPqK);
        asm volatile("cp.async.commit_group;");
    }
    if (tile_count > 2) {
        load_tile(2, 2 * kPqK);
        asm volatile("cp.async.commit_group;");
    }
    for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const int stage = static_cast<int>(tile_index % kPqStages);
        const int64_t remaining = tile_count - tile_index - 1;
        if (remaining >= 2) {
            asm volatile("cp.async.wait_group 2;");
        } else if (remaining == 1) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        // Barrier 1: every thread's cp.async for this stage is complete
        // before any thread reads tiles written by other threads.
        __syncthreads();

#pragma unroll
        for (int k_seg = 0; k_seg < kPqK / kMmaK; ++k_seg) {
            const int frag_col = thread_in_group * 4 + k_seg * 32;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int b_row = warp_n * 16 + nt * 8 + group;
                unsigned b_frag[2];
                b_frag[0] = *reinterpret_cast<const unsigned*>(
                    &b_tile[stage][b_row][frag_col]);
                b_frag[1] = *reinterpret_cast<const unsigned*>(
                    &b_tile[stage][b_row][frag_col + 16]);
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const int a_row0 = warp_m * 64 + mt * 16 + group;
                    unsigned a_frag[4];
                    a_frag[0] = *reinterpret_cast<const unsigned*>(
                        &a_tile[stage][a_row0][frag_col]);
                    a_frag[1] = *reinterpret_cast<const unsigned*>(
                        &a_tile[stage][a_row0 + 8][frag_col]);
                    a_frag[2] = *reinterpret_cast<const unsigned*>(
                        &a_tile[stage][a_row0][frag_col + 16]);
                    a_frag[3] = *reinterpret_cast<const unsigned*>(
                        &a_tile[stage][a_row0 + 8][frag_col + 16]);
                    mma_fp8_16832(acc + (nt * 4 + mt) * 4, a_frag, b_frag);
                }
            }
        }
        // Barrier 2: every thread finished reading this stage's tiles before
        // the prefetch for the (i+3)-th tile overwrites them.
        __syncthreads();
        if (tile_index + 3 < tile_count) {
            load_tile(stage, (tile_index + 3) * kPqK);
            asm volatile("cp.async.commit_group;");
        }
    }

    const float output_scale = scale * out_scale;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int64_t col = output_col + nt * 8;
        // Per-row store: FP8 packs two adjacent columns into one 16-bit
        // write; the BF16 path writes two scalars. Boundary columns fall
        // back to a scalar convert so the pack never crosses the row edge.
        auto store_out = [&](int64_t row, float v0, float v1) {
            if (row >= m) return;
            if constexpr (OutFp8) {
                if (col + 1 < n) {
                    *reinterpret_cast<unsigned short*>(
                        out_fp8 + row * n + col) =
                        static_cast<unsigned short>(__nv_cvt_float2_to_fp8x2(
                            make_float2(v0 * output_scale, v1 * output_scale),
                            __NV_SATFINITE, __NV_E4M3));
                } else {
                    out_fp8[row * n + col] = __nv_fp8_e4m3(v0 * output_scale);
                }
            } else {
                out_bf16[row * n + col] = __float2bfloat16(v0 * scale);
                if (col + 1 < n)
                    out_bf16[row * n + col + 1] = __float2bfloat16(v1 * scale);
            }
        };
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int64_t row0 = row_base + mt * 16;
            float* tile_acc = acc + (nt * 4 + mt) * 4;
            if (col < n) {
                store_out(row0, tile_acc[0], tile_acc[1]);
                store_out(row0 + 8, tile_acc[2], tile_acc[3]);
            }
        }
    }
}

template <bool AddBias = false, bool TrackAmax = true>
void launch_fused_fp8_gemm_fast(
    const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& out,
    const torch::Tensor& bias, const torch::Tensor& scale_a,
    const torch::Tensor& scale_b, torch::Tensor* amax_a,
    torch::Tensor* amax_b, int64_t m, int64_t n, int64_t k,
    cudaStream_t stream) {
    dim3 grid((n + kFastBlockN - 1) / kFastBlockN,
              (m + kFastBlockM - 1) / kFastBlockM);
    const auto* bias_ptr = AddBias
        ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr())
        : nullptr;
    auto kernel = fused_fp8_gemm_fast_kernel<AddBias, TrackAmax>;
    static bool attribute_set = false;
    if (!attribute_set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            kFastSmemBytes));
        attribute_set = true;
    }
    kernel<<<grid, kWarps * 32, kFastSmemBytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), bias_ptr,
        scale_a.data_ptr<float>(), scale_b.data_ptr<float>(),
        amax_a ? amax_a->data_ptr<float>() : nullptr,
        amax_b ? amax_b->data_ptr<float>() : nullptr, m, n, k);
}

void check_fp8_device(const torch::Tensor& tensor) {
    static std::mutex mutex;
    static std::unordered_map<int, bool> supported;
    const int device = tensor.device().index();
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto cached = supported.find(device);
        if (cached != supported.end()) {
            TORCH_CHECK(cached->second,
                        "fused FP8 MMA requires compute capability 8.9 or newer");
            return;
        }
    }

    const auto* properties = at::cuda::getDeviceProperties(device);
    const bool is_supported = properties->major > 8 ||
                              (properties->major == 8 && properties->minor >= 9);
    {
        std::lock_guard<std::mutex> lock(mutex);
        supported.emplace(device, is_supported);
    }
    TORCH_CHECK(is_supported,
                "fused FP8 MMA requires compute capability 8.9 or newer");
}

void check_scale(const torch::Tensor& scale, const torch::Tensor& input,
                 const char* name) {
    TORCH_CHECK(scale.is_cuda() && scale.device() == input.device() &&
                    scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
                name, " must be a CUDA float32 scalar on the input device");
}

}  // namespace

torch::Tensor fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor sx,
                     torch::Tensor sw) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16 &&
                    b.scalar_type() == torch::kBFloat16,
                "a and b must be bf16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.size(1) == b.size(1), "inner dim mismatch");
    check_scale(sx, a, "sx");
    check_scale(sw, a, "sw");
    check_fp8_device(a);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = torch::empty({a_c.size(0), b_c.size(0)}, a_c.options());
    torch::Tensor no_bias;
    launch_fused_fp8_gemm_fast<false, false>(
        a_c, b_c, out, no_bias, sx, sw, nullptr, nullptr,
        a_c.size(0), b_c.size(0), a_c.size(1), stream.stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor fp8_linear_forward_scaled(
    torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor sx,
    torch::Tensor sw, torch::Tensor sx_inv, torch::Tensor sw_inv,
    torch::Tensor amax_x, torch::Tensor amax_w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                    w.scalar_type() == torch::kBFloat16,
                "x and w must be bf16");
    TORCH_CHECK(x.device() == w.device(), "x and w must be on the same device");
    check_scale(sx, x, "sx");
    check_scale(sw, x, "sw");
    check_fp8_device(x);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_c = x.reshape({-1, w.size(1)}).contiguous();
    auto w_c = w.contiguous();
    int64_t m = x_c.size(0), k = x_c.size(1), n = w_c.size(0);
    TORCH_CHECK(w_c.dim() == 2 && w_c.size(1) == k, "inner dim mismatch");
    C10_CUDA_CHECK(cudaMemsetAsync(amax_x.data_ptr<float>(), 0, sizeof(float),
                                   stream.stream()));
    C10_CUDA_CHECK(cudaMemsetAsync(amax_w.data_ptr<float>(), 0, sizeof(float),
                                   stream.stream()));
    auto out = torch::empty({m, n}, x_c.options());
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda() && bias.device() == x.device() &&
                        bias.scalar_type() == torch::kBFloat16 &&
                        bias.numel() == n,
                    "bias must be CUDA bf16 with shape [N]");
        launch_fused_fp8_gemm_fast<true, true>(
            x_c, w_c, out, bias, sx, sw, &amax_x, &amax_w,
            m, n, k, stream.stream());
    } else {
        launch_fused_fp8_gemm_fast<false, true>(
            x_c, w_c, out, bias, sx, sw, &amax_x, &amax_w,
            m, n, k, stream.stream());
    }
    C10_CUDA_CHECK(cudaGetLastError());

    (void)sx_inv;
    (void)sw_inv;
    std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end() - 1);
    shape.push_back(n);
    return out.reshape(shape);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fp8_linear_backward_scaled(
    torch::Tensor g, torch::Tensor x, torch::Tensor w,
    std::vector<int64_t> masks, torch::Tensor sg, torch::Tensor sw,
    torch::Tensor sx, torch::Tensor sg_inv, torch::Tensor sw_inv,
    torch::Tensor sx_inv, torch::Tensor amax_g) {
    TORCH_CHECK(g.is_cuda() && x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16 &&
                    x.scalar_type() == torch::kBFloat16 &&
                    w.scalar_type() == torch::kBFloat16,
                "g, x, and w must be bf16");
    TORCH_CHECK(g.device() == x.device() && g.device() == w.device(),
                "g, x, and w must be on the same device");
    TORCH_CHECK(masks.size() == 3, "masks must contain three values");
    check_fp8_device(g);
    const at::cuda::OptionalCUDAGuard guard(g.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto g_c = g.reshape({-1, w.size(0)}).contiguous();
    auto x_c = x.reshape({-1, x.size(-1)}).contiguous();
    auto w_c = w.contiguous();
    int64_t m = g_c.size(0), n = w_c.size(0), k = w_c.size(1);
    TORCH_CHECK(x_c.size(0) == m && x_c.size(1) == k && g_c.size(1) == n,
                "backward shape mismatch");

    auto grad_input = torch::empty_like(x);
    auto grad_weight = torch::empty_like(w);
    auto grad_bias = torch::empty({0}, g.options());
    C10_CUDA_CHECK(cudaMemsetAsync(amax_g.data_ptr<float>(), 0, sizeof(float),
                                   stream.stream()));
    torch::Tensor no_bias;
    bool recorded_amax = false;
    if (masks[0]) {
        auto grad_input_2d = grad_input.reshape({m, k});
        // The fast kernel computes A @ B^T. A contiguous W^T makes dX use
        // the same coalesced forward tile path instead of scalar fragments.
        auto w_t = w_c.transpose(0, 1).contiguous();
        launch_fused_fp8_gemm_fast<false, true>(
            g_c, w_t, grad_input_2d, no_bias, sg, sw, &amax_g, nullptr,
            m, k, n, stream.stream());
        recorded_amax = true;
    }
    if (masks[1]) {
        // dW = G^T @ X, expressed as (G^T) @ (X^T)^T for the same kernel.
        auto g_t = g_c.transpose(0, 1).contiguous();
        auto x_t = x_c.transpose(0, 1).contiguous();
        if (recorded_amax) {
            launch_fused_fp8_gemm_fast<false, false>(
                g_t, x_t, grad_weight, no_bias, sg, sx, nullptr, nullptr,
                n, k, m, stream.stream());
        } else {
            launch_fused_fp8_gemm_fast<false, true>(
                g_t, x_t, grad_weight, no_bias, sg, sx, &amax_g, nullptr,
                n, k, m, stream.stream());
        }
        recorded_amax = true;
    }
    if (!recorded_amax) {
        amax_g.copy_(g_c.abs().amax().to(torch::kFloat32));
    }
    C10_CUDA_CHECK(cudaGetLastError());
    if (masks[2]) grad_bias = g_c.sum(0).to(g.scalar_type());

    (void)sg_inv;
    (void)sw_inv;
    (void)sx_inv;
    return {grad_input, grad_weight, grad_bias};
}

torch::Tensor fp8_mm_prequant(torch::Tensor a, torch::Tensor b,
                              torch::Tensor scale) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn &&
                    b.scalar_type() == torch::kFloat8_e4m3fn,
                "a and b must be fp8_e4m3fn");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.size(1) == b.size(1), "inner dim mismatch");
    check_scale(scale, a, "scale");
    check_fp8_device(a);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    int64_t m = a_c.size(0), k = a_c.size(1), n = b_c.size(0);
    auto out = torch::empty({m, n},
                            a_c.options().dtype(torch::kBFloat16));
    const float scale_value = scale.item<float>();
    dim3 grid((n + kPqBlockN - 1) / kPqBlockN,
              (m + kPqBlockM - 1) / kPqBlockM);
    fp8_mm_pq_kernel<false><<<grid, kWarps * 32, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(a_c.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(b_c.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nullptr,
        scale_value, 1.0f, m, n, k);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor fp8_mm_prequant_fp8(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor scale,
                                  torch::Tensor out_scale) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn &&
                    b.scalar_type() == torch::kFloat8_e4m3fn,
                "a and b must be fp8_e4m3fn");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.size(1) == b.size(1), "inner dim mismatch");
    check_scale(scale, a, "scale");
    check_scale(out_scale, a, "out_scale");
    check_fp8_device(a);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    int64_t m = a_c.size(0), k = a_c.size(1), n = b_c.size(0);
    auto out = torch::empty({m, n}, a_c.options());
    const float scale_value = scale.item<float>();
    const float out_scale_value = out_scale.item<float>();
    dim3 grid((n + kPqBlockN - 1) / kPqBlockN,
              (m + kPqBlockM - 1) / kPqBlockM);
    fp8_mm_pq_kernel<true><<<grid, kWarps * 32, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(a_c.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(b_c.data_ptr()), nullptr,
        reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr()),
        scale_value, out_scale_value, m, n, k);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_mm", &fp8_mm, py::arg("a"), py::arg("b"), py::arg("sx"),
          py::arg("sw"),
          "Fused BF16 input, E4M3 MMA, FP32 accumulation, BF16 output GEMM");
    m.def("fp8_mm_prequant", &fp8_mm_prequant, py::arg("a"), py::arg("b"),
          py::arg("scale"),
          "Pre-quantized FP8 GEMM with FP32 accumulation and BF16 output");
    m.def("fp8_mm_prequant_fp8", &fp8_mm_prequant_fp8, py::arg("a"),
          py::arg("b"), py::arg("scale"), py::arg("out_scale"),
          "Pre-quantized FP8 GEMM with FP32 accumulation and FP8 output");
    m.def("fp8_linear_forward_scaled", &fp8_linear_forward_scaled,
          py::arg("x"), py::arg("w"), py::arg("bias"), py::arg("sx"),
          py::arg("sw"), py::arg("sx_inv"), py::arg("sw_inv"),
          py::arg("amax_x"), py::arg("amax_w"),
          "Fused BF16-to-FP8 linear forward with FP32 accumulation");
    m.def("fp8_linear_backward_scaled", &fp8_linear_backward_scaled,
          py::arg("g"), py::arg("x"), py::arg("w"), py::arg("masks"),
          py::arg("sg"), py::arg("sw"), py::arg("sx"), py::arg("sg_inv"),
          py::arg("sw_inv"), py::arg("sx_inv"), py::arg("amax_g"),
          "Fused BF16-to-FP8 linear backward with FP32 accumulation");
}
