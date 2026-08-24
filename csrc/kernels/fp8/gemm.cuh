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
#include "../common/cp_async.cuh"
#include "../common/mma.cuh"
#include "../common/reduce.cuh"

namespace astrai {
namespace fp8 {

// m16n8k32 (see astrai::mma_shape<fp8 type>::k in common/mma.cuh)
constexpr int kMmaK = 32;
constexpr int kWarps = 8;  // 128x128 CTA = 8 warps

// log2 of a compile-time power of two (for tile_at's swizzle shift).
template <int N, int Acc = 0>
struct log2_const : log2_const<(N >> 1), Acc + 1> {};
template <int Acc>
struct log2_const<1, Acc> {
    static constexpr int value = Acc;
};

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
// warp_reduce_max / atomic_max_float (quantize amax) live in
// common/reduce.cuh; the cp.async pipeline primitives (predicated 16-byte
// copy, commit_group, wait_group + runtime dispatch) in common/cp_async.cuh.

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
// index is XORed with a row-dependent slice so a warp's fragment load (8
// consecutive rows x 16B) hits all 32 banks exactly once. With kChunks
// power-of-two chunks per row, the XOR source is the top log2(kChunks) bits
// of the row index within each group of 8:
//   kChunks=2 -> row bits [3]   (K=32:  rows r and r+4 diverge)
//   kChunks=4 -> row bits [2:1] (K=64:  rows diverge every 2)
//   kChunks=8 -> row bits [2:0] (K=128: every row)
// (row word-stride is K/4 words = 4*kChunks, so unswizzled rows r and
// r + 8/kChunks collide mod 32 banks; the XOR spreads the 8 rows of one
// ldmatrix matrix across the 8 distinct 4-bank groups.) Chunks stay
// contiguous, so the cp.async 16B staging path is unaffected.
template <int K, typename T8>
__device__ __forceinline__ T8* tile_at(T8* tile, int row, int col) {
    constexpr int kChunks = K / 16;  // 16B chunks per row
    static_assert(kChunks >= 1 && (kChunks & (kChunks - 1)) == 0,
                  "swizzle needs a power-of-two 16B-chunk count");
    constexpr int kShift = 3 - log2_const<kChunks>::value;
    return tile + row * K
           + ((((col >> 4) ^ ((row >> kShift) & (kChunks - 1))) << 4)
              + (col & 15));
}

// Stage-load one GEMM operand into the canonical flat [rows * K] shared tile
// (addressing via tile_at, so stores land in the swizzled layout). The
// transpose is folded into the staging step via a CUTLASS-style crosswise
// layout: RowMajor (stored [rows][contract]) copies 16-byte K-contiguous runs
// with cp.async, while ColMajor (stored [contract][rows]) reads 16-byte runs
// along the operand's contiguous non-contract dim and scatters them across
// the tile's rows. RowsTile is the tile's row capacity (kBlockM / kBlockN)
// and kThreads the CTA size; the runtime `rows` bound may be smaller (tail
// predication). `block_row` is this block's origin in the operand's row dim.
template <typename T8, int K, typename Layout, int RowsTile, int kThreads>
__device__ __forceinline__ void load_operand_tile(
    T8* tile, const T8* __restrict__ operand, int64_t rows,
    int64_t contract, int64_t ld, int tid, int64_t k_base,
    int64_t block_row) {
    constexpr int kChunks = K / 16;
    static_assert(RowsTile * kChunks % kThreads == 0,
                  "tile chunks must divide evenly across threads");
    constexpr int kCpt = RowsTile * kChunks / kThreads;  // chunks per thread
    if constexpr (std::is_same_v<Layout, ColMajor>) {
        // Operand stored [contract][rows]: contiguous along the non-contract
        // dim. Each thread scatters one 16-byte run per K/32 pass; when the
        // tile has more 16-row groups than warps (RowsTile > kThreads/2),
        // each thread covers several groups.
        constexpr int kWarpsTile = kThreads / 32;
        constexpr int kGroups = RowsTile / 16;
        static_assert(kGroups % kWarpsTile == 0,
                      "row groups must divide evenly across warps");
#pragma unroll
        for (int g = 0; g < kGroups / kWarpsTile; ++g) {
            const int rg = (tid >> 5) + g * kWarpsTile;
            const int kl = tid & 31;  // byte column within a 32B pass
            const int64_t r0 = block_row + rg * 16;
#pragma unroll
            for (int pass = 0; pass < K / 32; ++pass) {
                const int col = kl + pass * 32;
                const int64_t k_idx = k_base + col;
                const auto* src = operand + k_idx * ld + r0;
                if (k_idx < contract && r0 + 15 < rows &&
                    (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
                    const uint4 v = *reinterpret_cast<const uint4*>(src);
                    const auto* bytes = reinterpret_cast<const T8*>(&v);
                    // Scatter 16 bytes along the tile rows through tile_at's
                    // swizzle. Rows sharing a physical chunk form groups of
                    // (8 / kChunks) consecutive rows (see tile_at), so each
                    // group is one tile_at address plus a K-byte row stride.
                    constexpr int kGrp = 8 / kChunks;
#pragma unroll
                    for (int j = 0; j < 16 / kGrp; ++j) {
                        T8* p = tile_at<K>(tile,
                                           rg * 16 + j * kGrp, col);
#pragma unroll
                        for (int i = 0; i < kGrp; ++i)
                            p[i * K] = bytes[j * kGrp + i];
                    }
                } else {
                    // Predicated fallback: same layout, byte-granular gather.
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int64_t r_idx = r0 + i;
                        *tile_at<K>(tile, rg * 16 + i, col) =
                            (r_idx < rows && k_idx < contract)
                                ? operand[k_idx * ld + r_idx]
                                : T8(0.0f);
                    }
                }
            }
        }
    } else {
        // Operand stored [rows][contract]: contiguous along the contract dim.
        // Linear chunk mapping: thread covers kCpt consecutive 16B chunks of
        // one row (K=64: a contiguous 32B pair; K=32: a single chunk).
        const int r = tid / (kChunks / kCpt);
#pragma unroll
        for (int j = 0; j < kCpt; ++j) {
            const int c = ((tid % (kChunks / kCpt)) * kCpt + j) * 16;
            const int64_t row = block_row + r;
            const auto* src = operand + row * ld + k_base + c;
            T8* dst = tile_at<K>(tile, r, c);
            if (row < rows && k_base + c + 15 < contract &&
                (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
                astrai::cp_async_16(dst, src, true);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    dst[i] = row < rows && k_base + c + i < contract
                                 ? src[i]
                                 : T8(0.0f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-quantized GEMM kernel: FP8 A/B read straight into shared memory, FP32
// accumulation, BF16 or FP8 output. The input format follows Traits; the
// tile is compact (row = kK bytes) so MMA fragments read directly — no
// in-kernel transpose of the operands (the binding handles transposes).
// ---------------------------------------------------------------------------

// Swizzled 16B-chunk address (tile_at's layout) as a raw shared-memory
// pointer for ldmatrix. Valid for kK in {32, 64} (the swizzle itself lives
// only in tile_at; this wrapper just converts the element address).
template <typename T8, int kK>
__device__ __forceinline__ unsigned frag_addr(const T8* tile, int row,
                                              int chunk) {
    static_assert(kK == 32 || kK == 64,
                  "fragment swizzle offsets assume kK in {32, 64}");
    return __cvta_generic_to_shared(tile_at<kK>(tile, row, chunk << 4));
}

// LayoutA / LayoutB tag the operands' storage (CUTLASS-style, see common.h):
// A RowMajor = [M][K] / ColMajor = [K][M]; B RowMajor = [K][N] /
// ColMajor = [N][K]. The kernel always computes
//   out[m][n] = sum_p tileA[m][p] * tileB[n][p]
// with the tiles materialized in the canonical [M][kK] / [N][kK] layout, so the
// MMA fragments are read identically regardless of layout. The tags only
// change how the stage-load gathers the operand from global memory:
//   A ColMajor: tileA[m][p] = a[p*a_ld + m];  A RowMajor: a[m*a_ld + p]
//   B RowMajor: tileB[n][p] = b[p*b_ld + n];  B ColMajor: b[n*b_ld + p]
// BlockM x BlockN CTA as (BlockM/64) x (BlockN/32) warps of 64x32 warp tiles
// (mt x nt = 4x4 MMA each). The 64x128 variant runs 4 warps / 128 threads and
// exists for small-M calls: m <= 64 wastes half of every 128-row CTA, so the
// launcher dispatches to it there (see launch_fp8_gemm).
template <typename Traits, bool OutFp8 = false, typename LayoutA = RowMajor, typename LayoutB = RowMajor>
__global__ void __launch_bounds__((Traits::kBlockM / 64) * (Traits::kBlockN / 32) * 32, 2)
fp8_gemm_kernel(FP8Params p) {
    using T8 = std::conditional_t<Traits::kIsE5M2, __nv_fp8_e5m2, __nv_fp8_e4m3>;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kK = Traits::kK;
    constexpr int kStages = Traits::kStages;
    constexpr int kCtaThreads = (kBlockM / 64) * (kBlockN / 32) * 32;
    static_assert(kStages >= 1 && kStages <= 8,
                  "FP8 GEMM stages must be in the range [1, 8]");
    // Tiles are flat [rows * kK] with a 16B-chunk XOR swizzle (tile_at):
    // ldmatrix reads whole 16B chunks through the same mapping the staging
    // writes, and the swizzle removes the bank conflict the unswizzled
    // 8-word row stride caused (see tile_at).
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
    // L2-friendly rasterization (CUTLASS-style grouped launch order): remap
    // the linear block id so consecutive CTAs cover a group of kGroupM M-tiles
    // before advancing along N. All CTAs of one group share the same B column
    // stripe, so B tiles stay hot in L2 across the wave (the default
    // N-fastest order makes each wave touch every B tile instead).
    // Measured win for the A-crosswise layouts (10-21% at K>=2048) and loss
    // for A-congruous (-17..20%, A's cp.async stream prefers the N-fastest
    // order) — so the branch follows LayoutA.
    constexpr int kGroupM = 8;
    int block_m, block_n;
    if constexpr (std::is_same_v<LayoutA, ColMajor>) {
        const int blocks_m = gridDim.y;
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;
        const int group_first_m = (bid / (kGroupM * gridDim.x)) * kGroupM;
        const int group_rows =
            min(blocks_m - group_first_m, kGroupM);  // M-tail group is short
        block_m = group_first_m + bid % group_rows;
        block_n = (bid % (kGroupM * gridDim.x)) / group_rows;
    } else {
        block_m = blockIdx.y;
        block_n = blockIdx.x;
    }
    // 128x128 CTA = 8 warps as 2x4 warp tiles of 64x32 (mt x nt = 4x4 MMA).
    constexpr int warps_n = kBlockN / 32;
    const int warp_m = warp / warps_n;
    const int warp_n = warp % warps_n;
    const int64_t row_base =
        (int64_t)block_m * kBlockM + warp_m * 64 + group;
    const int64_t output_col =
        (int64_t)block_n * kBlockN + warp_n * 32 + thread_in_group * 2;
    const int a_row0 = warp_m * 64;  // + mt * 16 in the loop
    const int b_row0 = warp_n * 32;  // + nt * 8
    const float sa = *p.scale_a;
    const float sb = *p.scale_b;
    float acc[4][4][4] = {};  // [nt][mt][acc]

    // Both operands are staged into the canonical [M][kK] / [N][kK] shared
    // tiles regardless of their global layout (see load_operand_tile), so the
    // MMA fragment reads below stay unchanged across the four layout
    // combinations. A's tag already names the operand view ([M][K] =
    // [rows][contract]); B's tag is relative to the canonical [K][N], so the
    // stage-load sees its transpose (transpose_layout_t, see common.h).
    auto load_tile = [&](int stage, int64_t k_base) {
        load_operand_tile<T8, kK, LayoutA, kBlockM, kCtaThreads>(
            a_smem[stage], a, m, k, a_ld, tid, k_base,
            (int64_t)block_m * kBlockM);
        load_operand_tile<T8, kK, transpose_layout_t<LayoutB>, kBlockN,
                          kCtaThreads>(
            b_smem[stage], b, n, k, b_ld, tid, k_base,
            (int64_t)block_n * kBlockN);
    };

    const int64_t tile_count = (k + kK - 1) / kK;

    // Per-lane ldmatrix row/chunk selectors for common/mma.cuh's
    // ldmatrix_*_lane (the fragment tiles are XOR-swizzled per 16B chunk, so
    // each lane computes its own row/chunk address). Layout contract for fp8
    // m16n8k32 (values packed two-per-b16 slot, K-contiguous rows):
    //   x4 (A fragment): lane i points at tile row (i>>3 & 1)*8 + (i&7) of
    //       chunk (k_seg*2 + (i>>4)); reg j = matrix j = [row g][tig*4..+3] in
    //       the order (rows 0-7 c, rows 8-15 c, rows 0-7 c+1, rows 8-15 c+1) —
    //       exactly the mma.sync A operand layout.
    //   x2 (B fragment): lane i points at tile row (i&7) of chunk
    //       (k_seg*2 + ((i>>3) & 1)); reg j = [row(n) g][tig*4..+3] chunk c/c+1
    //       — exactly the mma.sync B operand layout (col operand, K-contiguous).
    const int r7 = lane & 7;          // row within the 8-row matrix
    const int rh8 = (lane >> 3) & 1;  // +8 rows (A: lanes 8-15, 24-31)
    const int rh16 = lane >> 4;       // +1 chunk (A: lanes 16-31; B uses rh8)

    // Prime the pipeline. Each committed group occupies one circular shared
    // memory stage; the loop also handles K dimensions smaller than kStages.
#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
        if (stage < tile_count) {
            load_tile(stage, static_cast<int64_t>(stage) * kK);
            astrai::cp_async_commit_group();
        }
    }

    for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const int stage = static_cast<int>(tile_index % kStages);
        const int64_t remaining = tile_count - tile_index - 1;

        // Keep up to kStages - 1 younger groups in flight while making the
        // oldest group (the current stage) ready for consumption.
        const int keep_groups =
            remaining < kStages - 1 ? static_cast<int>(remaining) : kStages - 1;
        astrai::cp_async_wait_group_dispatch<kStages - 1>(keep_groups);
        // Barrier 1: every thread's cp.async for this stage is complete
        // before any thread reads tiles written by other threads.
        __syncthreads();

        // 4 ldmatrix.x2 (B) + 4 ldmatrix.x4 (A) feed 16 mma.sync per k_seg —
        // 0.5 load instructions per MMA, versus 4.5 scalar LDS per MMA in
        // the 128x64-tile version (the kernel was LSU-issue-bound there).
        constexpr int kSegs = kK / kMmaK;
        // B fragments double-buffered across k_segs: the next k_seg's B load
        // is issued before the current k_seg's MMA sequence, so its LDS
        // latency hides behind the A pipeline + tensor-pipe work (same trick
        // as the A mt+1 prefetch below; costs kSegs x 8 registers).
        unsigned b_frag[2][4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int row = b_row0 + nt * 8 + r7;
            astrai::ldmatrix_x2_lane(b_frag[0][nt],
                    frag_addr<T8, kK>(b_smem[stage], row, rh8));
        }
#pragma unroll
        for (int k_seg = 0; k_seg < kSegs; ++k_seg) {
            const int bcur = k_seg & 1, bnext = bcur ^ 1;
            if (k_seg + 1 < kSegs) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int row = b_row0 + nt * 8 + r7;
                    astrai::ldmatrix_x2_lane(b_frag[bnext][nt],
                            frag_addr<T8, kK>(b_smem[stage], row,
                                              (k_seg + 1) * 2 + rh8));
                }
            }
            // Software-pipelined A fragments: the ldmatrix.x4 for row mt+1
            // is issued before the MMAs consuming row mt, so the LDS fixed
            // latency hides behind tensor-pipe work (cuts the `wait` stall,
            // ~2.3 cycles/issue before this). Costs 4 extra registers.
            unsigned a_frag[5][4];
            astrai::ldmatrix_x4_lane(a_frag[0],
                    frag_addr<T8, kK>(a_smem[stage], a_row0 + rh8 * 8 + r7,
                                      k_seg * 2 + rh16));
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                if (mt < 3)
                    astrai::ldmatrix_x4_lane(a_frag[mt + 1],
                            frag_addr<T8, kK>(
                                a_smem[stage],
                                a_row0 + (mt + 1) * 16 + rh8 * 8 + r7,
                                k_seg * 2 + rh16));
#pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    astrai::mma_sync<T8>(acc[nt][mt], a_frag[mt],
                                         b_frag[bcur][nt], acc[nt][mt]);
            }
        }
        // Barrier 2: every thread finished reading this stage's tiles before
        // the prefetch for the (i+kStages)-th tile overwrites them.
        __syncthreads();
        if (tile_index + kStages < tile_count) {
            load_tile(stage, (tile_index + kStages) * kK);
            astrai::cp_async_commit_group();
        }
    }

    const float output_scale = sa * sb;
    // Fused bias: BF16 raw values, or FP8 storage dequantized by its own
    // scale (bias_scale != null selects the FP8 path; the format follows the
    // kernel's Traits). Added in real units after the operand dequantization
    // and before any output quantization.
    const auto* bias16 = static_cast<const __nv_bfloat16*>(p.bias);
    const auto* bias8 = static_cast<const T8*>(p.bias);
    auto bias_val = [&](int64_t col) -> float {
        if (p.bias == nullptr || col >= n) return 0.0f;
        if (p.bias_scale == nullptr) return __bfloat162float(bias16[col]);
        return __half2float(__half(bias8[col])) * *p.bias_scale;
    };
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int64_t col = output_col + nt * 8;
        const float b0 = bias_val(col);
        const float b1 = bias_val(col + 1);
        // Per-row store: FP8 packs two adjacent columns into one 16-bit
        // write, BF16 into one 32-bit __nv_bfloat162 (single cvt+pack
        // instruction); boundary or unaligned columns fall back to scalar
        // converts so a pack never crosses the row edge or misaligns.
        auto store_out = [&](int64_t row, float v0, float v1) {
            if (row >= m) return;
            const float r0 = v0 * output_scale + b0;
            const float r1 = v1 * output_scale + b1;
            if constexpr (OutFp8) {
                if (col + 1 < n) {
                    *reinterpret_cast<unsigned short*>(out_fp8 + row * n + col) =
                        static_cast<unsigned short>(__nv_cvt_float2_to_fp8x2(
                            make_float2(r0 * *p.out_scale, r1 * *p.out_scale),
                            __NV_SATFINITE, __NV_E4M3));
                } else {
                    out_fp8[row * n + col] = __nv_fp8_e4m3(r0 * *p.out_scale);
                }
            } else {
                auto* dst = out_bf16 + row * n + col;
                if (col + 1 < n &&
                    (reinterpret_cast<uintptr_t>(dst) & 3) == 0) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) =
                        __floats2bfloat162_rn(r0, r1);
                } else {
                    dst[0] = __float2bfloat16(r0);
                    if (col + 1 < n) dst[1] = __float2bfloat16(r1);
                }
            }
        };
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int64_t row0 = row_base + mt * 16;
            float* tile_acc = acc[nt][mt];
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

// Pre-quantized GEMM tile config: 128x128 CTA (8 warps x 64x32 warp tiles).
// kK selects the K tile (32 or 64; 64 halves the __syncthreads count per K
// and doubles the MMA work per stage, at 2x the smem per stage — measured
// 10-35% across shapes, so 64 is the default). Stages=2 with kK=64 keeps the
// pipeline at 32KB smem; deeper pipelines only win on K >= 4096 squares and
// lose elsewhere. LayoutA/LayoutB mirror the kernel template (defaults keep
// the NN layout: out = a @ b). m <= 64 dispatches to the 64x128 CTA — a
// 128-row CTA would waste half its MMA work on predicated-off rows.
template <FP8Format Fmt, bool OutFp8 = false, typename LayoutA = RowMajor,
          typename LayoutB = RowMajor, int kK = 64, int Stages = 2>
void launch_fp8_gemm(const FP8Params& p, cudaStream_t stream) {
    dim3 grid((p.n + 127) / 128, (p.m + 127) / 128);
    if (p.m <= 64) {
        using Traits = Fp8GemmTraits<Fmt, 64, 128, kK, Stages>;
        fp8_gemm_kernel<Traits, OutFp8, LayoutA, LayoutB>
            <<<grid, (64 / 64) * (128 / 32) * 32, 0, stream>>>(p);
    } else {
        using Traits = Fp8GemmTraits<Fmt, 128, 128, kK, Stages>;
        fp8_gemm_kernel<Traits, OutFp8, LayoutA, LayoutB>
            <<<grid, (128 / 64) * (128 / 32) * 32, 0, stream>>>(p);
    }
}

}  // namespace fp8
}  // namespace astrai
