#pragma once
// FP8 GEMM device code — pure CUDA, no torch. Mirrors the attention kernel
// layout (attn_*_mma.cuh): kernels take the FP8Params POD, tile shape and
// FP8 format ride on compile-time template parameters, and launchers are
// plain functions usable from both the torch binding and pure C tests.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "common.h"
#include "../common/mma.cuh"

namespace fp8 {

// m16n8k32 (see astrai::mma_shape<fp8 type>::k in common/mma.cuh)
constexpr int kMmaK = 32;
constexpr int kWarps = 8;  // 128x64 CTA = 8 warps

// Map the FP8Format enum to the CUDA fp8 element type consumed by mma_sync.
template <FP8Format Fmt>
struct fp8_input {
    using type = __nv_fp8_e4m3;
};
template <>
struct fp8_input<FP8Format::E5M2> {
    using type = __nv_fp8_e5m2;
};

// ---------------------------------------------------------------------------
// Shared device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned pack_fp8x4_vector(
    float x0, float x1, float x2, float x3,
    __nv_fp8_interpretation_t fmt = __NV_E4M3) {
    const auto low = __nv_cvt_float2_to_fp8x2(make_float2(x0, x1),
                                              __NV_SATFINITE, fmt);
    const auto high = __nv_cvt_float2_to_fp8x2(make_float2(x2, x3),
                                               __NV_SATFINITE, fmt);
    return static_cast<unsigned>(low) | (static_cast<unsigned>(high) << 16);
}

// FP8 MMA lives in the shared astrai::mma_sync template (common/mma.cuh);
// instantiate it with fp8_input<Fmt>::type. Accumulates in-place: callers
// pass the same accumulator array as both `d` and `c`.

__device__ __forceinline__ void atomic_max_float(float* destination,
                                                  float value) {
    if (destination)
        atomicMax(reinterpret_cast<unsigned*>(destination),
                  __float_as_uint(value));
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

// One thread moves eight BF16 values (16 bytes) via cp.async; the uint4
// shape keeps source and destination naturally 128-bit aligned.
__device__ __forceinline__ void cp_async_bf16_8(
    __nv_bfloat16* destination, const __nv_bfloat16* source, bool valid) {
    const unsigned shared_address = __cvta_generic_to_shared(destination);
    const uint4* source_vec = reinterpret_cast<const uint4*>(source);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_address), "l"(source_vec),
                    "r"(valid ? 16 : 0));
}

// One thread moves sixteen FP8 values (16 bytes) via cp.async.
template <typename T>
__device__ __forceinline__ void cp_async_16b(T* destination,
                                             const T* source, bool valid) {
    const unsigned shared_address = __cvta_generic_to_shared(destination);
    const uint4* source_vec = reinterpret_cast<const uint4*>(source);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_address), "l"(source_vec),
                    "r"(valid ? 16 : 0));
}

// Convert four BF16 values to one 4xFP8 pack, tracking the raw (pre-scale)
// amax — scaling first would saturate amax at the FP8 max and collapse the
// scale. Format comes from Traits.
template <typename Traits, bool TrackAmax = true>
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
                             x2 * scale_inv, x3 * scale_inv,
                             Traits::kNvFormat);
}

// ---------------------------------------------------------------------------
// Quantize kernel: BF16 -> FP8 (E4M3 or E5M2), fused amax over raw values.
// ---------------------------------------------------------------------------

template <FP8Format Fmt>
__global__ void fp8_quantize_kernel(FP8Params p) {
    const float inv = 1.0f / *p.scale_a;
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(p.a_ptr);
    void* x8 = p.out_ptr;
    float* amax = p.amax_a;
    float local_amax = 0.0f;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < p.total;
         i += stride) {
        const float f = __bfloat162float(x[i]);
        local_amax = fmaxf(local_amax, fabsf(f));
        const float q = f * inv;
        if constexpr (Fmt == FP8Format::E5M2) {
            reinterpret_cast<__nv_fp8_e5m2*>(x8)[i] = __nv_fp8_e5m2(q);
        } else {
            reinterpret_cast<__nv_fp8_e4m3*>(x8)[i] = __nv_fp8_e4m3(q);
        }
    }
    if (amax) {
        local_amax = warp_reduce_max(local_amax);
        __shared__ float slots[32];
        if ((threadIdx.x & 31) == 0) slots[threadIdx.x >> 5] = local_amax;
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = 0.0f;
            for (int w = 0; w < (blockDim.x >> 5); ++w) v = fmaxf(v, slots[w]);
            atomic_max_float(amax, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused kernel: BF16 A/B -> inline E4M3 quantize -> ldmatrix fragments ->
// MMA -> BF16 out. 128x64 CTA / 64x16 warp tile / cp.async pipeline.
// The quantized FP8 tiles live in a separate smem region laid out around
// ldmatrix's single-address, 128-byte-strided matrices (16-byte rows):
//   A8: [M/16 block][4 sub-blocks of 8 rows x 16 fp8][...] where sub-block
//       order is (h0,m0-7), (h0,m8-15), (h1,m0-7), (h1,m8-15) — one
//       ldmatrix.x4 emits the whole m16n8k32 A fragment (regs 0..3 match).
//   B8: [N/8 block][2 sub-blocks of 8 rows x 16 fp8][...] with h0 then h1 —
//       one ldmatrix.x2 emits the m16n8k32 B fragment (regs 0,1).
// ---------------------------------------------------------------------------

template <typename Traits, bool AddBias, bool TrackAmax>
__global__ void fp8_fused_gemm_kernel(FP8Params p) {
    using T8 = __nv_fp8_e4m3;  // fused forward always quantizes to E4M3
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kK = Traits::kK;
    constexpr int kStages = Traits::kStages;
    constexpr int kWarpM = 64;  // warp tile rows (BlockM / 2)
    constexpr int kWarpN = 16;  // warp tile cols (BlockN / 4)
    constexpr int a_stride = kBlockM * kK;  // bf16 elements per A stage
    constexpr int b_stride = kBlockN * kK;  // bf16 elements per B stage
    // A8 block layout: (M/16) blocks x 4 sub-blocks x 128 B = BlockM*32 B.
    // B8 block layout: (N/8) blocks x 2 sub-blocks x 128 B = BlockN*32 B.
    constexpr int a8_bytes = kBlockM * 32;
    constexpr int b8_bytes = kBlockN * 32;
    // smem layout: [A bf16 stages][B bf16 stages][A8 fp8 tiles][B8 fp8 tiles]
    constexpr int bf16_bytes = kStages * (a_stride + b_stride) * 2;
    extern __shared__ char smem[];
    auto* a_bf16 = reinterpret_cast<__nv_bfloat16*>(smem);
    auto* b_bf16 = reinterpret_cast<__nv_bfloat16*>(smem + kStages * a_stride * 2);
    auto* a8 = reinterpret_cast<T8*>(smem + bf16_bytes);
    auto* b8 = reinterpret_cast<T8*>(smem + bf16_bytes + a8_bytes);
    __shared__ float warp_amax_a[kWarps];
    __shared__ float warp_amax_b[kWarps];

    const auto* a = reinterpret_cast<const __nv_bfloat16*>(p.a_ptr);
    const auto* b = reinterpret_cast<const __nv_bfloat16*>(p.b_ptr);
    auto* out = reinterpret_cast<__nv_bfloat16*>(p.out_ptr);
    const auto* bias = p.bias;
    const float* scale_a = p.scale_a;
    const float* scale_b = p.scale_b;
    float* amax_a = p.amax_a;
    float* amax_b = p.amax_b;
    const int64_t m = p.m, n = p.n, k = p.k;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    constexpr int warps_n = kBlockN / 16;
    const int warp_m = warp / warps_n;
    const int warp_n = warp % warps_n;
    const int64_t row_base = blockIdx.y * kBlockM + warp_m * kWarpM + group;
    const int64_t output_col =
        blockIdx.x * kBlockN + warp_n * 16 + thread_in_group * 2;
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
        for (int j = 0; j < kK / 32; ++j) {
            const int col = c0 + 32 * j;
            const bool full_chunk = k_base + col + 7 < k;
            const int64_t a_row = blockIdx.y * kBlockM + r0;
            const int64_t b_row = blockIdx.x * kBlockN + r0;
            auto* a_dst = &a_bf16[stage * a_stride + r0 * kK + col];
            auto* b_dst = &b_bf16[stage * b_stride + r0 * kK + col];
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
            if (r0 + kWarpM < kBlockM) {
                const int64_t a_row_hi = blockIdx.y * kBlockM + r0 + kWarpM;
                auto* a_dst_hi =
                    &a_bf16[stage * a_stride + (r0 + kWarpM) * kK + col];
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

    // Quantize the BF16 staging area into the ldmatrix-friendly FP8 tiles.
    // A8 sub-block for global row `row` and K half `h`:
    //   (row>>4)*512 + ((h<<1)|((row>>3)&1))*128 + (row&7)*16
    // B8 sub-block: (row>>3)*256 + h*128 + (row&7)*16.
    // Each thread emits one 4-FP8 pack at a time (256 threads, kK/4 = 8 packs
    // per row).
    auto quantize_tile = [&](int stage) {
        constexpr int kA_packs = kBlockM * kK / 4;
        constexpr int kB_packs = kBlockN * kK / 4;
#pragma unroll
        for (int i = tid; i < kA_packs; i += 256) {
            const int row = i >> 3;   // 8 packs per row
            const int k4 = (i & 7) * 4;
            const int half = k4 >> 4; // 0: k 0-15, 1: k 16-31
            const int k16 = k4 & 15;
            const int a8_idx =
                (row >> 4) * 512 + (((half << 1) | ((row >> 3) & 1)) * 128) +
                (row & 7) * 16 + k16;
            auto* src = &a_bf16[stage * a_stride + row * kK + k4];
            auto* dst = reinterpret_cast<unsigned*>(&a8[a8_idx]);
            *dst = load_fp8x4_from_bf16<Traits, TrackAmax>(
                src, inv_a, local_amax_a, track_amax_a);
        }
#pragma unroll
        for (int i = tid; i < kB_packs; i += 256) {
            const int row = i >> 3;
            const int k4 = (i & 7) * 4;
            const int half = k4 >> 4;
            const int k16 = k4 & 15;
            const int b8_idx =
                (row >> 3) * 256 + half * 128 + (row & 7) * 16 + k16;
            auto* src = &b_bf16[stage * b_stride + row * kK + k4];
            auto* dst = reinterpret_cast<unsigned*>(&b8[b8_idx]);
            *dst = load_fp8x4_from_bf16<Traits, TrackAmax>(
                src, inv_b, local_amax_b, track_amax_b);
        }
    };

    const int64_t tile_count = (k + kK - 1) / kK;
    load_tile(0, 0);
    asm volatile("cp.async.commit_group;");
    if (tile_count > 1) {
        load_tile(1, kK);
        asm volatile("cp.async.commit_group;");
    }
    if (tile_count > 2) {
        load_tile(2, 2 * kK);
        asm volatile("cp.async.commit_group;");
    }
    for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const int stage = static_cast<int>(tile_index % kStages);
        // 3-stage pipeline: at most 2 groups in flight; the tail of the K
        // loop waits for everything.
        const int64_t remaining = tile_count - tile_index - 1;
        if (remaining >= 2) {
            asm volatile("cp.async.wait_group 2;");
        } else if (remaining == 1) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        // wait_group only waits for this thread's async copies. All threads
        // must finish loading before the tile is read by the CTA.
        __syncthreads();
        quantize_tile(stage);
        __syncthreads();

        // kK == kMmaK, so one m16n8k32 MMA segment per K stage; fragments
        // come from the fp8 tiles via ldmatrix.
#pragma unroll
        for (int k_seg = 0; k_seg < kK / kMmaK; ++k_seg) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int b_row0 = warp_n * 16 + nt * 8;
                unsigned b_frag[2];
                // B8 block = (b_row0>>3), sub-blocks h0 then h1 at +0/+128.
                // ldmatrix: each thread supplies one matrix-row address —
                // threads 0-7 feed matrix 0 (h0) rows, 8-15 matrix 1 (h1);
                // the remaining threads' addresses are ignored.
                const int b8_base = (b_row0 >> 3) * 256;
                astrai::ldmatrix_x2<T8>(
                    b_frag,
                    &b8[b8_base + ((lane / 8) & 1) * 128 + (lane % 8) * 16]);
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const int a_row0 = warp_m * kWarpM + mt * 16;
                    unsigned a_frag[4];
                    // A8 block = (a_row0>>4); one x4 emits regs 0..3 in the
                    // exact mma A-operand order: h0m0-7, h0m8-15, h1m0-7,
                    // h1m8-15. Each thread supplies matrix (tid/8) row
                    // (tid%8) — all 32 addresses are used by x4.
                    const int a8_base = (a_row0 >> 4) * 512;
                    astrai::ldmatrix_x4<T8>(
                        a_frag,
                        &a8[a8_base + (lane / 8) * 128 + (lane % 8) * 16]);
                    astrai::mma_sync<typename fp8_input<Traits::kFormat>::type>(
                        acc + (nt * 4 + mt) * 4,
                        a_frag, b_frag, acc + (nt * 4 + mt) * 4);
                }
            }
        }
        __syncthreads();
        if (tile_index + 3 < tile_count) {
            load_tile(stage, (tile_index + 3) * kK);
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

// ---------------------------------------------------------------------------
// Pre-quantized kernel: FP8 A/B read straight into shared memory, FP32
// accumulation, BF16 or FP8 output. The input format follows Traits; the
// tile is compact (row = kK bytes) so MMA fragments read directly.
// ---------------------------------------------------------------------------

template <typename Traits, bool OutFp8 = false>
__global__ void fp8_pq_gemm_kernel(FP8Params p) {
    using T8 = std::conditional_t<Traits::kIsE5M2, __nv_fp8_e5m2, __nv_fp8_e4m3>;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kK = Traits::kK;
    constexpr int kStages = Traits::kStages;
    __shared__ __align__(16) T8 a_tile[kStages][kBlockM][kK];
    __shared__ __align__(16) T8 b_tile[kStages][kBlockN][kK];

    const auto* a = reinterpret_cast<const T8*>(p.a_ptr);
    const auto* b = reinterpret_cast<const T8*>(p.b_ptr);
    auto* out_bf16 = reinterpret_cast<__nv_bfloat16*>(p.out_ptr);
    auto* out_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(p.out_ptr);
    const int64_t m = p.m, n = p.n, k = p.k;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    constexpr int warps_n = kBlockN / 16;
    const int warp_m = warp / warps_n;
    const int warp_n = warp % warps_n;
    const int64_t row_base = blockIdx.y * kBlockM + warp_m * 64 + group;
    const int64_t output_col =
        blockIdx.x * kBlockN + warp_n * 16 + thread_in_group * 2;
    const float sa = *p.scale_a;
    const float sb = *p.scale_b;
    float acc[4 * 4 * 2] = {};

    // One A chunk (16 FP8) per thread covers the 128x32 tile; the first 128
    // threads issue the 64x32 B chunks.
    auto load_tile = [&](int stage, int64_t k_base) {
        const int r0 = tid >> 1;
        const int c0 = (tid & 1) * 16;
        const bool full_chunk = k_base + c0 + 15 < k;
        const int64_t a_row = blockIdx.y * kBlockM + r0;
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
                               : T8(0.0f);
            }
        }
        if (tid < 128) {
            const int64_t b_row = blockIdx.x * kBlockN + r0;
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
                                   : T8(0.0f);
                }
            }
        }
    };

    const int64_t tile_count = (k + kK - 1) / kK;
    load_tile(0, 0);
    asm volatile("cp.async.commit_group;");
    if (tile_count > 1) {
        load_tile(1, kK);
        asm volatile("cp.async.commit_group;");
    }
    if (tile_count > 2) {
        load_tile(2, 2 * kK);
        asm volatile("cp.async.commit_group;");
    }
    for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const int stage = static_cast<int>(tile_index % kStages);
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
        for (int k_seg = 0; k_seg < kK / kMmaK; ++k_seg) {
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
                    astrai::mma_sync<typename fp8_input<Traits::kFormat>::type>(
                        acc + (nt * 4 + mt) * 4,
                        a_frag, b_frag, acc + (nt * 4 + mt) * 4);
                }
            }
        }
        // Barrier 2: every thread finished reading this stage's tiles before
        // the prefetch for the (i+3)-th tile overwrites them.
        __syncthreads();
        if (tile_index + 3 < tile_count) {
            load_tile(stage, (tile_index + 3) * kK);
            asm volatile("cp.async.commit_group;");
        }
    }

    const float output_scale = sa * sb;
    const float o8_scale = OutFp8 ? output_scale * *p.out_scale : 0.0f;
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
                    *reinterpret_cast<unsigned short*>(out_fp8 + row * n + col) =
                        static_cast<unsigned short>(__nv_cvt_float2_to_fp8x2(
                            make_float2(v0 * o8_scale, v1 * o8_scale),
                            __NV_SATFINITE, __NV_E4M3));
                } else {
                    out_fp8[row * n + col] = __nv_fp8_e4m3(v0 * o8_scale);
                }
            } else {
                out_bf16[row * n + col] = __float2bfloat16(v0 * output_scale);
                if (col + 1 < n)
                    out_bf16[row * n + col + 1] =
                        __float2bfloat16(v1 * output_scale);
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

// ---------------------------------------------------------------------------
// Launchers — pure CUDA (no torch), usable from the binding and pure C tests.
// ---------------------------------------------------------------------------

// Fused forward tile config: 128x64 CTA, K=32, 3-stage cp.async pipeline,
// plus the fp8 ldmatrix tile region (A8[2][BlockM][16] + B8[2][BlockN][16]).
using FusedTraits = Fp8GemmTraits<FP8Format::E4M3, 128, 64, 32, 3>;
// Pre-quantized tile config: 128x64 CTA, K=32, 3-stage pipeline.
template <FP8Format Fmt>
using PqTraits = Fp8GemmTraits<Fmt, 128, 64, 32, 3>;

template <FP8Format Fmt>
void launch_fp8_quantize(const FP8Params& p, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int64_t blocks = (p.total + kThreads - 1) / kThreads;
    fp8_quantize_kernel<Fmt><<<blocks, kThreads, 0, stream>>>(p);
}

template <bool AddBias, bool TrackAmax>
void launch_fp8_fused(const FP8Params& p, cudaStream_t stream) {
    // bf16 staging (3 stages) + fp8 ldmatrix tiles (A8[2][M][16] + B8[2][N][16])
    constexpr int kSmemBytes =
        FusedTraits::kStages *
            (FusedTraits::kBlockM * FusedTraits::kK +
             FusedTraits::kBlockN * FusedTraits::kK) *
            2 +
        2 * FusedTraits::kBlockM * 16 + 2 * FusedTraits::kBlockN * 16;
    dim3 grid((p.n + FusedTraits::kBlockN - 1) / FusedTraits::kBlockN,
              (p.m + FusedTraits::kBlockM - 1) / FusedTraits::kBlockM);
    auto kernel = fp8_fused_gemm_kernel<FusedTraits, AddBias, TrackAmax>;
    static bool attribute_set = false;
    if (!attribute_set) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        attribute_set = true;
    }
    kernel<<<grid, kWarps * 32, kSmemBytes, stream>>>(p);
}

template <FP8Format Fmt, bool OutFp8 = false>
void launch_fp8_pq(const FP8Params& p, cudaStream_t stream) {
    using Traits = PqTraits<Fmt>;
    dim3 grid((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
              (p.m + Traits::kBlockM - 1) / Traits::kBlockM);
    fp8_pq_gemm_kernel<Traits, OutFp8><<<grid, kWarps * 32, 0, stream>>>(p);
}

}  // namespace fp8
