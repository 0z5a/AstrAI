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

// PTX requires wait_group's operand to be an immediate value. Keep it as a
// template argument so the stage policy remains compile-time configurable.
template <int KeepGroups>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7,
                  "cp.async.wait_group supports immediates in [0, 7]");
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
}

template <int MaxKeepGroups>
__device__ __forceinline__ void cp_async_wait_group_dispatch(int keep_groups) {
    static_assert(MaxKeepGroups >= 0 && MaxKeepGroups <= 7,
                  "cp.async.wait_group supports immediates in [0, 7]");
    if (keep_groups == MaxKeepGroups) {
        cp_async_wait_group<MaxKeepGroups>();
    } else if constexpr (MaxKeepGroups > 0) {
        cp_async_wait_group_dispatch<MaxKeepGroups - 1>(keep_groups);
    } else {
        cp_async_wait_group<0>();
    }
}

template <int Stages>
__device__ __forceinline__ void cp_async_commit_group() {
    static_assert(Stages >= 1 && Stages <= 8,
                  "FP8 GEMM stages must be in the range [1, 8]");
    asm volatile("cp.async.commit_group;");
}

// ---------------------------------------------------------------------------
// Quantize kernel: BF16 -> FP8 (E4M3 or E5M2), fused amax over raw values.
// ---------------------------------------------------------------------------

// Convert one packed bf16 pair to one packed fp8 pair. amax sees the *raw*
// (unscaled) values; the stored bytes see value * inv. Bit-identical to the
// scalar __nv_fp8_*(q) constructor path (round-nearest-even + satfinite).
template <FP8Format Fmt>
__device__ __forceinline__ unsigned quantize2(unsigned pair, float inv,
                                              float& amax) {
    const float lo = __bfloat162float(__ushort_as_bfloat16(pair & 0xffffu));
    const float hi = __bfloat162float(__ushort_as_bfloat16(pair >> 16));
    amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    constexpr __nv_fp8_interpretation_t kFmt =
        Fmt == FP8Format::E5M2 ? __NV_E5M2 : __NV_E4M3;
    return static_cast<unsigned>(
        __nv_cvt_float2_to_fp8x2(make_float2(lo * inv, hi * inv),
                                 __NV_SATFINITE, kFmt));
}

template <FP8Format Fmt>
__global__ void fp8_quantize_kernel(FP8Params p) {
    const float inv = 1.0f / *p.scale_a;
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(p.a_ptr);
    void* x8 = p.out_ptr;
    float* amax = p.amax_a;
    float local_amax = 0.0f;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // Vectorized body: 8 bf16 (16B load) -> 8 fp8 (8B store) per step. Torch
    // allocations are >=16B aligned and the binding passes freshly allocated
    // contiguous buffers, so element 0 keeps the uint4/uint2 accesses
    // natural; a misaligned base (contiguous view with an odd storage
    // offset) falls back to the scalar loop below via total_vec = 0.
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(x8))
         & 15) == 0;
    const int64_t total_vec = aligned ? p.total / 8 : 0;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    uint2* o8 = reinterpret_cast<uint2*>(x8);
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec;
         i += stride) {
        const uint4 v = xv[i];
        const unsigned pair[4] = {v.x, v.y, v.z, v.w};
        unsigned packed[2] = {0u, 0u};
#pragma unroll
        for (int j = 0; j < 4; ++j)
            packed[j >> 1] |= quantize2<Fmt>(pair[j], inv, local_amax)
                              << (16 * (j & 1));
        o8[i] = make_uint2(packed[0], packed[1]);
    }
    // Scalar tail (and full fallback for misaligned bases).
    for (int64_t i = total_vec * 8 + blockIdx.x * blockDim.x + threadIdx.x;
         i < p.total; i += stride) {
        const float f = __bfloat162float(x[i]);
        local_amax = fmaxf(local_amax, fabsf(f));
        if constexpr (Fmt == FP8Format::E5M2) {
            reinterpret_cast<__nv_fp8_e5m2*>(x8)[i] = __nv_fp8_e5m2(f * inv);
        } else {
            reinterpret_cast<__nv_fp8_e4m3*>(x8)[i] = __nv_fp8_e4m3(f * inv);
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

// Swizzled address inside a flat [rows * K] staging tile: the 16-byte chunk
// index is XORed with row bits starting at bit 2. Unswizzled, a kK=32 row
// spans only 8 words, so a warp's fragment load (8 consecutive rows x 4B,
// e.g. a_row0+0..7) maps rows r and r+4 onto the same banks — a 2-way
// conflict on every LDS. XORing the chunk index with row bit 2 shifts rows
// 4..7 by one chunk so each warp's 32-word read hits all 32 banks exactly
// once. Chunks stay contiguous, so the cp.async 16B staging path is
// unaffected. Validated for kK=32 (2 chunks); larger power-of-two chunk
// counts compile but need their own bank analysis.
template <int K, typename T8>
__device__ __forceinline__ T8* tile_at(T8* tile, int row, int col) {
    constexpr int kChunks = K / 16;  // 16B chunks per row
    static_assert(kChunks >= 1 && (kChunks & (kChunks - 1)) == 0,
                  "swizzle needs a power-of-two 16B-chunk count");
    return tile + row * K
           + ((((col >> 4) ^ ((row >> 2) & (kChunks - 1))) << 4)
              + (col & 15));
}

// Stage-load one GEMM operand into the canonical flat [rows * K] shared tile
// (addressing via tile_at, so stores land in the swizzled layout). The
// transpose is folded into the staging step via a CUTLASS-style crosswise
// layout: the congruous case copies 16-byte K-contiguous runs with cp.async,
// while the transposed case reads 16-byte runs along the operand's contiguous
// (non-contract) dim and scatters them across the tile's rows. `block_row` is
// this block's origin in the operand's row dim; the caller restricts which
// threads invoke it (all threads for A, the first 128 for B).
template <typename T8, int K, bool Trans>
__device__ __forceinline__ void load_operand_tile(
    T8* tile, const T8* __restrict__ operand, int64_t rows,
    int64_t contract, int64_t ld, int tid, int64_t k_base,
    int64_t block_row) {
    if constexpr (Trans) {
        // Operand stored [contract][rows]: contiguous along the non-contract dim.
        const int rg = tid >> 5;  // Rows / 16 row-groups
        const int kl = tid & 31;  // K lanes
        const int64_t k_idx = k_base + kl;
        const int64_t r0 = block_row + rg * 16;
        const auto* src = operand + k_idx * ld + r0;
        const bool aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
        if (k_idx < contract && r0 + 15 < rows && aligned) {
            const uint4 v = *reinterpret_cast<const uint4*>(src);
            const auto* bytes = reinterpret_cast<const T8*>(&v);
            // Scatter 16 bytes along the tile rows. The swizzle bit flips
            // every 4 rows ((rg*16 + i) >> 2 & 1 == (i >> 2) & 1), and the
            // physical column of row group g is kl ^ (16 * (g & 1)) — so the
            // whole 16-byte scatter is one base pointer plus two alternating
            // column offsets, no per-byte XOR in the address math.
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                T8* p = tile + (rg * 16 + 4 * g) * K
                        + (g & 1 ? (kl ^ 16) : kl);
                p[0] = bytes[4 * g];
                p[K] = bytes[4 * g + 1];
                p[2 * K] = bytes[4 * g + 2];
                p[3 * K] = bytes[4 * g + 3];
            }
        } else {
            // Predicated fallback: same layout, byte-granular gather.
            const int col = kl;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const T8* src_g = operand + k_idx * ld + r0 + 4 * g;
                T8* p = tile + (rg * 16 + 4 * g) * K
                        + (g & 1 ? (col ^ 16) : col);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int64_t r_idx = r0 + 4 * g + i;
                    p[i * K] = (r_idx < rows && k_idx < contract)
                                   ? src_g[i]
                                   : T8(0.0f);
                }
            }
        }
    } else {
        // Operand stored [rows][contract]: contiguous along the contract dim.
        const int r = tid >> 1;
        const int c = (tid & 1) * 16;
        const int64_t row = block_row + r;
        // c is a multiple of 16, so the whole 16-byte run shares one chunk
        // and dst[i] addressing below matches tile_at<K>(tile, r, c + i).
        T8* dst = tile_at<K>(tile, r, c);
        const auto* src = operand + row * ld + k_base + c;
        const bool full = k_base + c + 15 < contract;
        if (row < rows && full &&
            (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
            cp_async_16b(dst, src, true);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                dst[i] = row < rows && k_base + c + i < contract ? src[i]
                                                                 : T8(0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-quantized GEMM kernel: FP8 A/B read straight into shared memory, FP32
// accumulation, BF16 or FP8 output. The input format follows Traits; the
// tile is compact (row = kK bytes) so MMA fragments read directly — no
// in-kernel transpose of the operands (the binding handles transposes).
// ---------------------------------------------------------------------------

// TransA / TransB select the operand memory layout. The kernel always computes
//   out[m][n] = sum_p tileA[m][p] * tileB[n][p]
// with the tiles materialized in the canonical [M][kK] / [N][kK] layout, so the
// MMA fragments are read identically regardless of layout. The two flags only
// change how the stage-load gathers the operand from global memory:
//   TransA: tileA[m][p] = a[p*a_ld + m]  (A stored [K][M], i.e. A^T)
//           else        a[m*a_ld + p]   (A stored [M][K])
//   TransB: tileB[n][p] = b[n*b_ld + p]  (B stored [N][K])
//           else        b[p*b_ld + n]   (B stored [K][N], read transposed)
template <typename Traits, bool OutFp8 = false, bool TransA = false, bool TransB = true>
__global__ void fp8_gemm_kernel(FP8Params p) {
    using T8 = std::conditional_t<Traits::kIsE5M2, __nv_fp8_e5m2, __nv_fp8_e4m3>;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kK = Traits::kK;
    constexpr int kStages = Traits::kStages;
    static_assert(kStages >= 1 && kStages <= 8,
                  "FP8 GEMM stages must be in the range [1, 8]");
    // Tiles are flat [rows * kK] with a 16B-chunk XOR swizzle (tile_at): the
    // fragments read 4-byte K-contiguous chunks through the same mapping the
    // staging writes, and the swizzle removes the 2-way bank conflict the
    // unswizzled 8-word row stride caused (see tile_at).
    __shared__ __align__(16) T8 a_smem[kStages][kBlockM * kK];
    __shared__ __align__(16) T8 b_smem[kStages][kBlockN * kK];

    const auto* a = reinterpret_cast<const T8*>(p.a_ptr);
    const auto* b = reinterpret_cast<const T8*>(p.b_ptr);
    auto* out_bf16 = reinterpret_cast<__nv_bfloat16*>(p.out_ptr);
    auto* out_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(p.out_ptr);
    const int64_t m = p.m, n = p.n, k = p.k;
    const int64_t a_ld = p.a_ld, b_ld = p.b_ld;

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

    // Both operands are staged into the canonical [M][kK] / [N][kK] shared
    // tiles regardless of their global layout (see load_operand_tile), so the
    // MMA fragment reads below stay unchanged across the four layout flags.
    auto load_tile = [&](int stage, int64_t k_base) {
        // load_operand_tile's `Trans` means "the operand's contiguous dim is
        // the non-contract dim" (crosswise load). For A that is TransA; for B
        // the storage flag is inverted (TransB=true stores B as [N][K], i.e.
        // K-contiguous, which is the congruous case).
        load_operand_tile<T8, kK, TransA>(
            a_smem[stage], a, m, k, a_ld, tid, k_base, blockIdx.y * kBlockM);
        if (tid < 128)
            load_operand_tile<T8, kK, !TransB>(
                b_smem[stage], b, n, k, b_ld, tid, k_base,
                blockIdx.x * kBlockN);
    };

    const int64_t tile_count = (k + kK - 1) / kK;

    // Hoisted swizzle offsets for the fragment reads (see tile_at). Every
    // row this thread reads — B rows warp_n*16 + nt*8 + group and A rows
    // warp_m*64 + mt*16 + group (± 8) — has swizzle bit (row >> 2) & 1
    // equal to (group >> 2) & 1: all other terms (16, 32, 64 row offsets)
    // shift in multiples of 4 rows and leave bit 2 of the row untouched.
    // The two chunk halves of a fragment differ by exactly one chunk bit,
    // so the high-half offset is 16 - low. Net effect: the hot loop pays
    // one add per LDS, same as the unswizzled layout.
    static_assert(kK == 32,
                  "hoisted fragment-swizzle offsets assume kK == 32");
    const int tig4 = thread_in_group * 4;
    const int sw_lo = ((group >> 2) & 1) << 4;
    const int sw_hi = 16 - sw_lo;

    // Prime the pipeline. Each committed group occupies one circular shared
    // memory stage; the loop also handles K dimensions smaller than kStages.
#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
        if (stage < tile_count) {
            load_tile(stage, static_cast<int64_t>(stage) * kK);
            cp_async_commit_group<kStages>();
        }
    }

    for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const int stage = static_cast<int>(tile_index % kStages);
        const int64_t remaining = tile_count - tile_index - 1;

        // Keep up to kStages - 1 younger groups in flight while making the
        // oldest group (the current stage) ready for consumption.
        const int keep_groups =
            remaining < kStages - 1 ? static_cast<int>(remaining) : kStages - 1;
        cp_async_wait_group_dispatch<kStages - 1>(keep_groups);
        // Barrier 1: every thread's cp.async for this stage is complete
        // before any thread reads tiles written by other threads.
        __syncthreads();

#pragma unroll
        for (int k_seg = 0; k_seg < kK / kMmaK; ++k_seg) {
            const int klo = k_seg * kMmaK + tig4;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const T8* brow =
                    b_smem[stage] + (warp_n * 16 + nt * 8 + group) * kK;
                // B fragment: two 4-FP8 chunks (K-contiguous) at output row.
                unsigned b_frag[2];
                b_frag[0] = *reinterpret_cast<const unsigned*>(brow + klo + sw_lo);
                b_frag[1] = *reinterpret_cast<const unsigned*>(brow + klo + sw_hi);
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const T8* arow =
                        a_smem[stage] + (warp_m * 64 + mt * 16 + group) * kK;
                    unsigned a_frag[4];
                    a_frag[0] = *reinterpret_cast<const unsigned*>(arow + klo + sw_lo);
                    a_frag[1] = *reinterpret_cast<const unsigned*>(
                        arow + 8 * kK + klo + sw_lo);
                    a_frag[2] = *reinterpret_cast<const unsigned*>(arow + klo + sw_hi);
                    a_frag[3] = *reinterpret_cast<const unsigned*>(
                        arow + 8 * kK + klo + sw_hi);
                    astrai::mma_sync<typename fp8_input<Traits::kFormat>::type>(
                        acc + (nt * 4 + mt) * 4,
                        a_frag, b_frag, acc + (nt * 4 + mt) * 4);
                }
            }
        }
        // Barrier 2: every thread finished reading this stage's tiles before
        // the prefetch for the (i+3)-th tile overwrites them.
        __syncthreads();
        if (tile_index + kStages < tile_count) {
            load_tile(stage, (tile_index + kStages) * kK);
            cp_async_commit_group<kStages>();
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
    // One block per 256 vectors (8 elements each); at least one block so the
    // scalar tail of a tiny / misaligned tensor is still covered.
    int64_t blocks = (p.total / 8 + kThreads - 1) / kThreads;
    if (blocks < 1) blocks = 1;
    fp8_quantize_kernel<Fmt><<<blocks, kThreads, 0, stream>>>(p);
}

// Pre-quantized GEMM tile config: 128x64 CTA, K=32, 2-stage pipeline by
// default. Stages remains an explicit template override for tuning.
// TransA/TransB mirror the kernel template (defaults keep the NT layout:
// out = a @ b^T with both operands K-contiguous).
template <FP8Format Fmt, bool OutFp8 = false, bool TransA = false,
          bool TransB = true, int Stages = 2>
void launch_fp8_gemm(const FP8Params& p, cudaStream_t stream) {
    using Traits = Fp8GemmTraits<Fmt, 128, 64, 32, Stages>;
    dim3 grid((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
              (p.m + Traits::kBlockM - 1) / Traits::kBlockM);
    fp8_gemm_kernel<Traits, OutFp8, TransA, TransB><<<grid, kWarps * 32, 0, stream>>>(p);
}

}  // namespace fp8
