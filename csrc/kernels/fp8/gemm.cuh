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
// Pre-quantized GEMM kernel: FP8 A/B read straight into shared memory, FP32
// accumulation, BF16 or FP8 output. The input format follows Traits; the
// tile is compact (row = kK bytes) so MMA fragments read directly — no
// in-kernel transpose of the operands (the binding handles transposes).
// ---------------------------------------------------------------------------

template <typename Traits, bool OutFp8 = false>
__global__ void fp8_gemm_kernel(FP8Params p) {
    using T8 = std::conditional_t<Traits::kIsE5M2, __nv_fp8_e5m2, __nv_fp8_e4m3>;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kK = Traits::kK;
    constexpr int kStages = Traits::kStages;
    // Tiles are [M][kK] / [N][kK]: each row is kK bytes (16B-aligned for
    // cp.async), and the MMA fragments read 4-byte-aligned K-contiguous
    // chunks directly from them.
    __shared__ __align__(16) T8 a_smem[kStages][kBlockM][kK];
    __shared__ __align__(16) T8 b_smem[kStages][kBlockN][kK];

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
    // threads issue the 64x32 B chunks. Both operands are already in the MMA
    // row-major / col-major layout ([M][K] with K contiguous), so each thread
    // copies a 16-byte-aligned run straight into the tile via cp.async.
    auto load_tile = [&](int stage, int64_t k_base) {
        const int r0 = tid >> 1;
        const int c0 = (tid & 1) * 16;
        const bool full_chunk = k_base + c0 + 15 < k;
        const int64_t a_row = blockIdx.y * kBlockM + r0;
        auto* a_dst = &a_smem[stage][r0][c0];
        const auto* a_ptr = a + a_row * k + k_base + c0;
        const bool full_a = a_row < m && full_chunk;
        const bool aligned_a =
            (reinterpret_cast<uintptr_t>(a_ptr) & 15) == 0;
        if (full_a && aligned_a) {
            cp_async_16b(a_dst, a_ptr, true);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                a_dst[i] = a_row < m && k_base + c0 + i < k
                               ? a_ptr[i]
                               : T8(0.0f);
        }
        if (tid < 128) {
            const int b_row = blockIdx.x * kBlockN + r0;
            auto* b_dst = &b_smem[stage][r0][c0];
            const auto* b_ptr = b + b_row * k + k_base + c0;
            const bool full_b = b_row < n && full_chunk;
            const bool aligned_b =
                (reinterpret_cast<uintptr_t>(b_ptr) & 15) == 0;
            if (full_b && aligned_b) {
                cp_async_16b(b_dst, b_ptr, true);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    b_dst[i] = b_row < n && k_base + c0 + i < k
                                   ? b_ptr[i]
                                   : T8(0.0f);
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
                // B fragment: two 4-FP8 chunks (K-contiguous) at output row.
                unsigned b_frag[2];
                b_frag[0] = *reinterpret_cast<const unsigned*>(
                    &b_smem[stage][b_row][frag_col]);
                b_frag[1] = *reinterpret_cast<const unsigned*>(
                    &b_smem[stage][b_row][frag_col + 16]);
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const int a_row0 = warp_m * 64 + mt * 16 + group;
                    unsigned a_frag[4];
                    a_frag[0] = *reinterpret_cast<const unsigned*>(
                        &a_smem[stage][a_row0][frag_col]);
                    a_frag[1] = *reinterpret_cast<const unsigned*>(
                        &a_smem[stage][a_row0 + 8][frag_col]);
                    a_frag[2] = *reinterpret_cast<const unsigned*>(
                        &a_smem[stage][a_row0][frag_col + 16]);
                    a_frag[3] = *reinterpret_cast<const unsigned*>(
                        &a_smem[stage][a_row0 + 8][frag_col + 16]);
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

template <FP8Format Fmt>
void launch_fp8_quantize(const FP8Params& p, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int64_t blocks = (p.total + kThreads - 1) / kThreads;
    fp8_quantize_kernel<Fmt><<<blocks, kThreads, 0, stream>>>(p);
}

// Pre-quantized GEMM tile config: 128x64 CTA, K=32, 3-stage pipeline.
template <FP8Format Fmt, bool OutFp8 = false>
void launch_fp8_gemm(const FP8Params& p, cudaStream_t stream) {
    using Traits = Fp8GemmTraits<Fmt, 128, 64, 32, 3>;
    dim3 grid((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
              (p.m + Traits::kBlockM - 1) / Traits::kBlockM);
    fp8_gemm_kernel<Traits, OutFp8><<<grid, kWarps * 32, 0, stream>>>(p);
}

}  // namespace fp8
