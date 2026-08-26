#pragma once
// FP8 GEMM device code — pure CUDA, no torch. Mirrors the attention kernel
// layout (attn_*_mma.cuh): kernels take the FP8Params POD, tile shape and
// FP8 format ride on compile-time template parameters, and launchers are
// plain functions usable from both the torch binding and pure C tests. The
// quantize kernel lives in quantize.cuh.

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

// log2 of a compile-time power of two (for tile_at's swizzle shift).
template <int N, int Acc = 0>
struct log2_const : log2_const<(N >> 1), Acc + 1> {};
template <int Acc>
struct log2_const<1, Acc> {
    static constexpr int value = Acc;
};

// ---------------------------------------------------------------------------
// Shared device helpers
// ---------------------------------------------------------------------------

// FP8 MMA lives in the shared astrai::mma_sync template (common/mma.cuh);
// instantiate it with the kernel's T8. Accumulates in-place: callers pass
// the same accumulator array as both `d` and `c`.
// The cp.async pipeline primitives (predicated 16-byte copy, commit_group,
// wait_group + runtime dispatch) live in common/cp_async.cuh.

// ---------------------------------------------------------------------------

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
    return tile + row * K +
           ((((col >> 4) ^ ((row >> kShift) & (kChunks - 1))) << 4) + (col & 15));
}

// Stage-load a CONGRUOUS operand (stored [rows][contract], contract-
// contiguous — the only cp.async-able shape for the canonical tile) into the
// flat [rows * K] shared tile via tile_at's swizzle. Crosswise operands go
// through load_crosswise_direct instead.
template <typename T8, int K, int RowsTile, int kThreads>
__device__ __forceinline__ void
load_operand_tile(T8* tile, const T8* __restrict__ operand, int64_t rows,
                  int64_t contract, int64_t ld, int tid, int64_t k_base,
                  int64_t block_row) {
    constexpr int kChunks = K / 16;
    static_assert(RowsTile * kChunks % kThreads == 0,
                  "tile chunks must divide evenly across threads");
    constexpr int kCpt = RowsTile * kChunks / kThreads;  // chunks per thread
    // Linear chunk mapping: thread covers kCpt consecutive 16B chunks of
    // one row (K=64: a contiguous 32B pair; K=32: a single chunk).
    constexpr int kCpr = kChunks / kCpt;  // chunks per row slice
    const int r = tid / kCpr;
    const int c0 = (tid % kCpr) * kCpt * 16;
    const int64_t row = block_row + r;
    const bool row_ok = row < rows;
    // k_base and every c are multiples of 16, so the per-chunk sources
    // share the row base's alignment.
    const auto* src = operand + row * ld + k_base;
    const bool chunk_aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
#pragma unroll
    for (int j = 0; j < kCpt; ++j) {
        const int c = c0 + j * 16;
        T8* dst = tile_at<K>(tile, r, c);
        if (row_ok && chunk_aligned && k_base + c + 15 < contract) {
            astrai::cp_async_16(dst, src + c);
        } else {
            // Tail chunk (or misaligned base): predicated scalar fill.
#pragma unroll
            for (int i = 0; i < 16; ++i)
                dst[i] =
                    row_ok && k_base + c + i < contract ? src[c + i] : T8(0.0f);
        }
    }
}

// Interior-tile congruous load: zero predication. Valid when
// block_row + RowsTile <= rows, k_base + K <= contract and
// (operand base | ld | k_base) is 16B-aligned — the kernel's fast_cta peel
// guarantees all three. With n = a thread's first chunk a multiple of kCpt,
// (n+j)^swz == (n^swz)^j, so the swizzled destination of chunk j is the
// base pointer XOR (j << 4): the whole address math folds into one
// immediate XOR per chunk (~3 inst/chunk vs ~9 predicated).
template <typename T8, int K, int RowsTile, int kThreads>
__device__ __forceinline__ void
load_operand_tile_interior(T8* tile, const T8* __restrict__ operand,
                           int64_t ld, int tid, int64_t k_base,
                           int64_t block_row) {
    constexpr int kChunks = K / 16;
    static_assert(RowsTile * kChunks % kThreads == 0,
                  "tile chunks must divide evenly across threads");
    constexpr int kCpt = RowsTile * kChunks / kThreads;
    constexpr int kCpr = kChunks / kCpt;
    const int r = tid / kCpr;
    const int c0 = (tid % kCpr) * kCpt * 16;
    const char* src = reinterpret_cast<const char*>(
        operand + (block_row + r) * ld + k_base + c0);
    const uintptr_t dst = reinterpret_cast<uintptr_t>(tile_at<K>(tile, r, c0));
#pragma unroll
    for (int j = 0; j < kCpt; ++j)
        astrai::cp_async_16(reinterpret_cast<T8*>(dst ^ (j << 4)),
                            src + j * 16);
}

// Loop-carried prefetch state for one congruous operand ring (perf 6.2):
// per-thread (r, c0) of the interior copy — the same mapping
// load_operand_tile_interior uses — with the swizzled stage destination and
// the global source pointer both carried across k-tiles, so each prefetch
// chunk is one LDGSTS at [wr ^ (j << 4)] / [src + j*16] issued straight from
// registers.
//
// Whether an operand has a carry is a property of its layout, so the guard
// lives in the type: the false specialization (crosswise operand — direct
// LDG+PRMT staging, no cp.async) is an empty no-op. Crosswise kernel
// instantiations therefore compile no dead declarations and use sites need
// no `if constexpr` and no [[maybe_unused]].
template <bool kAsync, typename T8, int kK, int kRowsTile, int kThreads>
struct PrefetchCarry;

template <typename T8, int kK, int kRowsTile, int kThreads>
struct PrefetchCarry<true, T8, kK, kRowsTile, kThreads> {
    static constexpr int kCpt = kRowsTile * (kK / 16) / kThreads;
    static constexpr int kCpr = (kK / 16) / kCpt;
    unsigned wr = 0;    // current stage's swizzled destination offset
    unsigned wr0 = 0;   // slot-0 wrap base
    unsigned wrEnd = 0; // one-past-the-ring sentinel
    const char* src = nullptr;  // current tile's global source bytes

    __device__ __forceinline__ PrefetchCarry(
        const T8* ring, int ringSlots, int stageElems, const T8* operand,
        int64_t ld, int64_t blockRow, int tid, int firstTile) {
        const int r = tid / kCpr;
        const int c0 = (tid % kCpr) * kCpt * 16;
        const T8* slot0 = ring + (firstTile % ringSlots) * stageElems;
        const unsigned laneOff = static_cast<unsigned>(
            (const char*)tile_at<kK>(slot0, r, c0) - (const char*)slot0);
        const unsigned base = __cvta_generic_to_shared(ring) + laneOff;
        wr = base + (unsigned)((firstTile % ringSlots) * stageElems);
        wr0 = base;
        wrEnd = base + (unsigned)(ringSlots * stageElems);
        src = reinterpret_cast<const char*>(
                  operand + (blockRow + r) * ld + c0) +
              (int64_t)firstTile * kK;
    }

    // Emit this thread's chunks for the current tile. `pf` false (loop
    // tail) zero-fills: src_size=0 reads nothing, and the destination is
    // the slot compute(i-1) already released.
    __device__ __forceinline__ void emit(bool pf) const {
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(wr ^ (unsigned)(j << 4), src + j * 16, pf);
    }

    __device__ __forceinline__ void advance(int stageElems) {
        wr += (unsigned)stageElems;
        if (wr == wrEnd) wr = wr0;
        src += kK;
    }
};

template <typename T8, int kK, int kRowsTile, int kThreads>
struct PrefetchCarry<false, T8, kK, kRowsTile, kThreads> {
    __device__ __forceinline__ PrefetchCarry(
        const T8*, int, int, const T8*, int64_t, int64_t, int, int) {}
    __device__ __forceinline__ void emit(bool) const {}
    __device__ __forceinline__ void advance(int) {}
};

// ---------------------------------------------------------------------------
// Pre-quantized GEMM kernel: FP8 A/B read straight into shared memory, FP32
// accumulation, BF16 or FP8 output. The input format follows Traits; the
// tile is compact (row = kK bytes) so MMA fragments read directly — no
// in-kernel transpose of the operands (the binding handles transposes).
// ---------------------------------------------------------------------------

// Direct (synchronous) crosswise load into a canonical rotating stage:
// LDG.128 x4 (4 consecutive contract bytes x 16 rows) + in-register PRMT
// transpose + 16 STS.32. Crosswise operands cannot cp.async into the
// canonical [rows][contract] tile (a 16B global run holds one contract byte
// for each of 16 rows), so they take this path. A staged variant
// (cp.async into K-major staging + per-tile smem->smem transpose) measured
// 15-20% SLOWER than this direct load across every probed shape, including
// DRAM-streaming B operands — see git history (5745c2f) if it ever needs
// revisiting for other SKUs.
template <typename T8, int K, int RowsTile, int kThreads>
__device__ __forceinline__ void
load_crosswise_direct(T8* tile, const T8* __restrict__ operand, int64_t rows,
                      int64_t contract, int64_t ld, int tid, int64_t k_base,
                      int64_t block_row) {
    constexpr int kQuads = K / 4;    // 4-byte contract quads per tile
    constexpr int kGroups = RowsTile / 16;
    constexpr int kTChunks = kQuads * kGroups;  // 64B chunks per tile
    // r0 is always a multiple of 16 (block_row is a multiple of RowsTile and
    // each group covers 16 rows), and p*ld keeps the base 16B-aligned
    // whenever ld is, so every run of a chunk shares one alignment verdict.
    const bool run_aligned =
        ((reinterpret_cast<uintptr_t>(operand) | ld) & 15) == 0;
    for (int chunk = tid; chunk < kTChunks; chunk += kThreads) {
        const int quad = chunk / kGroups;
        const int rg = chunk % kGroups;
        const int64_t r0 = block_row + rg * 16;
        const bool rows_full = r0 + 15 < rows;
        if (rows_full && run_aligned) {
            const int64_t p0 = k_base + quad * 4;
            uint4 v[4];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                // Contract tail: a run past k carries zero bytes; they flow
                // through the PRMT transpose like any other value.
                if (p0 + s < contract)
                    v[s] = *reinterpret_cast<const uint4*>(
                        operand + (p0 + s) * ld + r0);
                else
                    v[s] = make_uint4(0u, 0u, 0u, 0u);
            }
            const unsigned* bytes = reinterpret_cast<const unsigned*>(v);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                // word i = row r0+i's quad: byte i of each of the four runs
                // [v0.b(i), v1.b(i), v2.b(i), v3.b(i)]. Byte i of a uint4
                // lives in its (i>>2)-th 32-bit register.
                const unsigned nib = i & 3;
                const unsigned sel = nib | ((nib + 4) << 4);
                const unsigned w01 =
                    __byte_perm(bytes[0 + (i >> 2)], bytes[4 + (i >> 2)], sel);
                const unsigned w23 =
                    __byte_perm(bytes[8 + (i >> 2)], bytes[12 + (i >> 2)], sel);
                *reinterpret_cast<unsigned*>(tile_at<K>(tile, rg * 16 + i,
                                                        quad * 4)) =
                    __byte_perm(w01, w23, 0x5410u);
            }
        } else {
            // Row-tail or misaligned chunk: byte-granular gather with
            // per-row predication; contract-tail columns zero-fill.
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int col = quad * 4 + s;
                if (k_base + col >= contract) {
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        *tile_at<K>(tile, rg * 16 + i, col) = T8(0.0f);
                    continue;
                }
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int64_t r_idx = r0 + i;
                    *tile_at<K>(tile, rg * 16 + i, col) =
                        r_idx < rows
                            ? operand[(k_base + col) * ld + r_idx]
                            : T8(0.0f);
                }
            }
        }
    }
}

// Layout-aware shared-memory budget and occupancy hint. Canonic rings hold
// kStages+1 buffers (LeanRing=false): the load for tile i+kStages targets
// slot (i-1)%(kStages+1) — already consumed — so the pure-congruous path
// needs no post-compute barrier (one __syncthreads per k-tile). LeanRing
// keeps the ring at kStages buffers for small CTAs whose occupancy comes
// from more resident CTAs (less smem) rather than a deeper rotation; it
// brings back barrier 4.
// The 48KB static-smem watermark picks the resident-CTA hint for
// __launch_bounds__ (sm_89: 100KB smem per SM, so two CTAs fit while each
// stays within the static budget).
template <typename Traits, typename LayoutA, typename LayoutB,
          bool LeanRing = false>
struct Fp8GemmSmem {
    // Crosswise (direct-load) operands: A ColMajor storage, B RowMajor
    // storage (B's tag is relative to the canonical [K][N]).
    static constexpr bool kDirectA = std::is_same_v<LayoutA, ColMajor>;
    static constexpr bool kDirectB = std::is_same_v<LayoutB, RowMajor>;
    // LeanRing shrinks only the congruous (async) operand rings; a direct
    // operand's ring stays kStages+1 deep (see the kernel's ring note).
    static constexpr int kARing = kDirectA ? Traits::kStages + 1
                                           : Traits::kStages + !LeanRing;
    static constexpr int kBRing = kDirectB ? Traits::kStages + 1
                                           : Traits::kStages + !LeanRing;
    static constexpr int kBytes =
        kARing * Traits::kBlockM * Traits::kK +
        kBRing * Traits::kBlockN * Traits::kK;
    static constexpr int kMinCtas = kBytes <= 48 * 1024 ? 2 : 1;
};

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
template <typename Traits, typename LayoutA = RowMajor, typename LayoutB = RowMajor, int kRasterGroup = 0,
        bool kLeanRing = false, bool kStreamOut = false,
        bool kFastLoop = false>
__global__ void __launch_bounds__(Traits::kCtaThreads,
                                  Fp8GemmSmem<Traits, LayoutA, LayoutB,
                                              kLeanRing>::kMinCtas)
    fp8_gemm_kernel(FP8Params p) {
    using T8 = std::conditional_t<Traits::kIsE5M2, __nv_fp8_e5m2, __nv_fp8_e4m3>;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kK = Traits::kK;
    constexpr int kStages = Traits::kStages;
    constexpr int kCtaThreads = Traits::kCtaThreads;
    constexpr bool kDirectA =
        Fp8GemmSmem<Traits, LayoutA, LayoutB, kLeanRing>::kDirectA;
    constexpr bool kDirectB =
        Fp8GemmSmem<Traits, LayoutA, LayoutB, kLeanRing>::kDirectB;
    static_assert(kStages >= 1 && kStages <= 8,
                  "FP8 GEMM stages must be in [1, 8]");
    // Tiles are flat [rows * kK] with a 16B-chunk XOR swizzle (tile_at):
    // ldmatrix reads whole 16B chunks through the same mapping the staging
    // writes, and the swizzle removes the bank conflict the unswizzled
    // 8-word row stride caused (see tile_at). The stages live in dynamic
    // shared memory so deep pipelines (kStages * (kBlockM + kBlockN) * kK >
    // 48KB static limit) opt in via cudaFuncSetAttribute in the launcher.
    extern __shared__ __align__(16) char fp8_gemm_smem[];
    // Per operand: congruous = kStages+1 rotating canonical buffers — the
    // load for tile i+kStages targets slot (i-1)%(kStages+1), which compute
    // finished reading before this iteration's barrier 1, so NO post-compute
    // barrier is needed on the pure-congruous path (one __syncthreads per
    // k-tile, the classic multistage rotation); direct-crosswise rotates the
    // same kStages+1 ring for the same reason.
    constexpr int kAStageBytes = kBlockM * kK;
    constexpr int kBStageBytes = kBlockN * kK;
    // Direct-crosswise operands always rotate kStages+1 buffers: their
    // prefetch issues right after barrier 1 (targeting the slot compute(i-1)
    // released), so a kStages-deep lean ring would race the in-flight MMA
    // reads. The lean ring applies only to congruous operands, whose cp.async
    // prefetch sits behind the restored barrier 4.
    constexpr int kARing = kDirectA ? kStages + 1 : kStages + !kLeanRing;
    constexpr int kBRing = kDirectB ? kStages + 1 : kStages + !kLeanRing;
    T8* const a_base = reinterpret_cast<T8*>(fp8_gemm_smem);
    T8* const b_base =
        reinterpret_cast<T8*>(fp8_gemm_smem + kARing * kAStageBytes);

    // Batch slice (grid.z): broadcast operands carry a 0 stride, so the
    // same pointer serves every batch.
    const auto* a = reinterpret_cast<const T8*>(p.a_ptr) +
                    (int64_t)blockIdx.z * p.a_batch_stride;
    const auto* b = reinterpret_cast<const T8*>(p.b_ptr) +
                    (int64_t)blockIdx.z * p.b_batch_stride;
    auto* out_bf16 = reinterpret_cast<__nv_bfloat16*>(p.out_ptr) +
                     (int64_t)blockIdx.z * p.out_batch_stride;
    const int64_t m = p.m, n = p.n, k = p.k;
    const int64_t a_ld = p.a_ld, b_ld = p.b_ld;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    // Tile scheduler: the linear CTA id maps to (block_m, block_n) in
    // grouped (L2-friendly, CUTLASS-style) or plain raster order — the
    // grouped order makes consecutive CTAs cover a group of kRasterGroup
    // M-tiles before advancing along N, so all CTAs of one group share the
    // same B column stripe and B tiles stay hot in L2 across the wave (the
    // plain N-fastest order makes each wave touch every B tile instead;
    // kRasterGroup=0 selects plain, the measured best for dX's crosswise-B
    // layouts where grouping measured neutral).
    // Persistent schedules (static round-robin and an atomic ticket
    // dispenser, grid capped at the resident CTAs) were both measured and
    // rejected on L20: the stride desynchronizes the in-flight window
    // (-4..-8%), and the ticket variant recovers the L2 locality but lands
    // within noise of plain waves (its loop-head barrier costs what the
    // CTA-restart overlap saves). Keep the classic retiring-wave launch.
    int block_m, block_n;
    if constexpr (kRasterGroup > 0) {
        constexpr int kGroupM = kRasterGroup;
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
    // CTA = (BlockM/WarpM) x (BlockN/WarpN) warps of WarpM x WarpN tiles,
    // each warp computing (WarpM/16) x (WarpN/8) m16n8k32 MMAs (mt x nt).
    // The default 128x128 CTA runs 8 warps of 64x32 (mt x nt = 4x4); the
    // small-shape path uses 64x64 CTAs of 32x32 warps (cuBLAS-style) so more
    // CTAs fit per SM (see launch_fp8_gemm).
    constexpr int kMt = Traits::kWarpM / 16;  // 16-row MMA tiles per warp
    constexpr int kNt = Traits::kWarpN / 8;   // 8-col MMA tiles per warp
    constexpr int kSegs = kK / kMmaK;         // mma-sized k segments per tile
    const int warp_m = warp / Traits::kWarpsN;
    const int warp_n = warp % Traits::kWarpsN;
    const int a_row0 = warp_m * Traits::kWarpM;  // + mt * 16 in the loop
    const int b_row0 = warp_n * Traits::kWarpN;  // + nt * 8
    const float scale = *p.scale;
    float acc[kNt][kMt][4] = {};  // [nt][mt][acc]

    // Both operands end up in the canonical [M][kK] / [N][kK] shared tiles
    // the MMA fragments read, regardless of their global layout. A's tag
    // already names the operand view ([M][K] = [rows][contract]); B's tag is
    // relative to the canonical [K][N], so the stage-load sees its transpose
    // (transpose_layout_t, see common.h). Congruous operands cp.async
    // straight into their rotating canonical buffers; crosswise operands
    // take load_direct's LDG+PRMT path below.
    // Stage-slot helpers: the rings rotate one slot per k-tile, so callers
    // either compute the slot from the tile index (prologue, generic loop)
    // or carry an advancing pointer (steady-state fast loop below).
    auto a_stage_of = [&](int64_t tile) -> T8* {
        return a_base + (size_t)(tile % kARing) * kAStageBytes;
    };
    auto b_stage_of = [&](int64_t tile) -> T8* {
        return b_base + (size_t)(tile % kBRing) * kBStageBytes;
    };
    // Asynchronous loads for tile `tile`: congruous operands cp.async into
    // their canonical rings. Called after the post-compute barrier, alongside
    // the commit.
    auto load_async = [&](T8* a_stage, T8* b_stage, int64_t k_base) {
        if constexpr (!kDirectA)
            load_operand_tile<T8, kK, kBlockM, kCtaThreads>(
                a_stage, a, m, k, a_ld, tid, k_base, (int64_t)block_m * kBlockM);
        if constexpr (!kDirectB)
            load_operand_tile<T8, kK, kBlockN, kCtaThreads>(
                b_stage, b, n, k, b_ld, tid, k_base,
                (int64_t)block_n * kBlockN);
    };
    // Predication-free interior variant of load_async: congruous operands
    // with full CTA rows, aligned (base | ld), k_base + kK <= k. fast_cta
    // admits only congruous operands, so no crosswise fallback is needed.
    auto load_async_fast = [&](T8* a_stage, T8* b_stage, int64_t k_base) {
        if constexpr (!kDirectA)
            load_operand_tile_interior<T8, kK, kBlockM, kCtaThreads>(
                a_stage, a, a_ld, tid, k_base, (int64_t)block_m * kBlockM);
        if constexpr (!kDirectB)
            load_operand_tile_interior<T8, kK, kBlockN, kCtaThreads>(
                b_stage, b, b_ld, tid, k_base, (int64_t)block_n * kBlockN);
    };
    // Synchronous direct-crosswise loads for tile `tile` into the operand's
    // (kStages+1)-deep canonical ring. In the steady state this runs right
    // after barrier 1, so the LDG latency and the PRMT transpose overlap the
    // MMA phase of the current tile instead of stalling the inter-barrier
    // window (which dominated the dX/dW stall profile: barrier 3.7-4.1 +
    // long-scoreboard 1.6-1.8 stalls per issue on the production shapes).
    // Ring safety: the write targets buffer (i+kStages)%(kStages+1) =
    // (i-1)%(kStages+1), which compute(i-1) finished reading before the
    // previous barrier and compute(i+kStages) does not touch until several
    // barriers later.
    auto load_direct = [&](T8* a_stage, T8* b_stage, int64_t k_base) {
        if constexpr (kDirectA)
            load_crosswise_direct<T8, kK, kBlockM, kCtaThreads>(
                a_stage, a, m, k, a_ld, tid, k_base,
                (int64_t)block_m * kBlockM);
        if constexpr (kDirectB)
            load_crosswise_direct<T8, kK, kBlockN, kCtaThreads>(
                b_stage, b, n, k, b_ld, tid, k_base,
                (int64_t)block_n * kBlockN);
    };

    const int64_t tile_count = (k + kK - 1) / kK;
    // Interior-CTA peel (kFastLoop instantiations only): when both operands
    // are congruous, whole-CTA, 16B-aligned and K has no tail, the mainloop
    // runs a compile-time-specialized copy whose loads carry no predication
    // — the per-chunk guards cost ~6 of ~100 instructions per warp per
    // k-tile, and the small-CTA path is issue-bound there (measured
    // +4.5..10% on 256³..1024³; the 128x128 kernel regressed ~3% with the
    // same change, so only the small CTA opts in). All verdicts are uniform
    // per CTA: one branch picks the loop copy.
    const bool fast_cta =
        kFastLoop && !kDirectA && !kDirectB &&
        ((int64_t)block_m * kBlockM + kBlockM <= m) &&
        ((int64_t)block_n * kBlockN + kBlockN <= n) &&
        ((reinterpret_cast<uintptr_t>(a) | (uint64_t)a_ld) & 15) == 0 &&
        ((reinterpret_cast<uintptr_t>(b) | (uint64_t)b_ld) & 15) == 0 &&
        (k % kK) == 0;

    // Per-lane ldmatrix fragment addressing (base-pair scheme, mirrored from
    // the cuBLAS SASS: one base register per operand per k_seg, every
    // fragment offset an LDSM immediate — zero address arithmetic inside the
    // MMA phase). The closure works because the XOR swizzle's source bits
    // come only from the lane's row-within-matrix (r7): the 8- and 16-row
    // fragment steps (nt*8, mt*16) never reach them, so
    //   addr(s, mt) = lane_base + mt*(16*kK)  ^ (s<<5)     [A, x4 fragment]
    //   addr(s, nt) = lane_base + nt*(8*kK)   ^ (s<<5)     [B, x2 fragment]
    // where the ^ (s<<5) lands inside the 16B-chunk swizzle field (each k_seg
    // advances the chunk index by 2 = 32B) and the step lands outside it.
    //   kChunks=4 (K=64):  swizzle bits = row[2:1] = r7[2:1]
    //   kChunks=8 (K=128): swizzle bits = row[2:0] = r7[2:0]
    //   kChunks=2 (K=32):  swizzle bit  = row[2]   = r7[2]  (single k_seg)
    // Replaces the former a_off[kSegs][kMt]/b_off[kSegs][kNt] runtime tables
    // (16 registers + one IADD per LDSM): at 131 regs the tables spilled and
    // ptxas rematerialized every address each k-tile (~55 of 146 hot-loop
    // instructions were LOP3/IMAD address math; cuBLAS's inner loop has ~0).
    const int r7 = lane & 7;          // row within the 8-row matrix
    const int rh8 = (lane >> 3) & 1;  // +8 rows (A: lanes 8-15, 24-31)
    const int rh16 = lane >> 4;       // +1 chunk (A: lanes 16-31; B uses rh8)
    constexpr int kChunks = kK / 16;
    constexpr int kShift = 3 - log2_const<kChunks>::value;  // tile_at's shift
    const unsigned lswz =
        static_cast<unsigned>((r7 >> kShift) & (kChunks - 1));
    constexpr unsigned kMtStep = 16 * kK;  // bytes per m-tile row step
    constexpr unsigned kNtStep = 8 * kK;   // bytes per n-tile row step
    constexpr unsigned kSegXor = 32;       // chunk-index +2 per k_seg
    // Stage-relative, loop-invariant per-lane bases (added to each ring
    // slot's converted base once per k-tile). A's fragment row carries the
    // +8-row half (rh8) and the +1-chunk half (rh16); B's carries rh8 as its
    // chunk half — matching the m16n8k32 operand layouts above.
    const unsigned a_lane_off = static_cast<unsigned>(
        (a_row0 + rh8 * 8 + r7) * kK + ((rh16 ^ lswz) << 4));
    const unsigned b_lane_off =
        static_cast<unsigned>((b_row0 + r7) * kK + ((rh8 ^ lswz) << 4));
    // x4-paired B loads (cuBLAS/CUTLASS loop shape): one ldmatrix.x4 feeds
    // the two adjacent nt fragments — 2 x4 instead of 4 x2 per k_seg (12
    // LDSM per k-tile instead of 16). Lane contract: lanes 0-7 address
    // rows n0..n7 chunk c, lanes 8-15 rows n0..n7 chunk c+1, lanes 16-23
    // rows n8..n15 chunk c, lanes 24-31 rows n8..n15 chunk c+1; regs
    // {r0,r1} are the even nt's k-halves, {r2,r3} the odd nt's. The +8-row
    // step never reaches the swizzle source bits for kK <= 64 (kChunks<=4:
    // bits row[2:1]), so lanes 16-31 reuse the same lswz and each pair
    // address is the even-nt base + p*(16*kK). kK=128 swizzles on row[2:0]
    // where +8 flips bits — that config keeps the x2 loads.
    constexpr bool kPairB = kK / 16 <= 4;
    static_assert(!kPairB || kNt % 2 == 0, "B pairing needs even kNt");
    constexpr unsigned kPairStep = 16 * kK;  // bytes per nt-pair row step
    const unsigned b4_lane_off = b_lane_off + rh16 * kPairStep / 2;

    // Prime the pipeline. Each committed group occupies one circular shared
    // memory stage. The commit is unconditional: when K is shorter than the
    // pipeline (tile_count < kStages) the skipped stages commit empty groups
    // so the group sequence stays tile-indexed — the steady-state
    // wait_group<kStages-1> below is then correct for every iteration and
    // no runtime wait-count dispatch is needed (the dispatch ladder cost 16
    // instructions per k-tile: ISETP/SEL chains picking DEPBAR immediates).
    // Direct loads run synchronously here (back to back with their commit);
    // the steady state below overlaps them with the compute phase.
#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
        if (stage < tile_count) {
            if (fast_cta)
                load_async_fast(a_stage_of(stage), b_stage_of(stage),
                                (int64_t)stage * kK);
            else
                load_async(a_stage_of(stage), b_stage_of(stage),
                           (int64_t)stage * kK);
            load_direct(a_stage_of(stage), b_stage_of(stage),
                        (int64_t)stage * kK);
        }
        astrai::cp_async_commit_group();
    }

    // Steady-state read carries: the LDSM base of the current k-tile's
    // stage with the lane offset folded in, advanced one stage per
    // iteration with an equality wrap (the add sequence is exact). This
    // replaces the per-k-tile (tile % ring) * stage_bytes recomputation —
    // its SASS form was a UIMAD.WIDE magic-division ladder, ~10
    // uniform-pipe instructions per operand per k-tile (perf 6.2).
    const unsigned a_rd0 = __cvta_generic_to_shared(a_base) + a_lane_off;
    const unsigned b_rd0 = __cvta_generic_to_shared(b_base) +
                           (kPairB ? b4_lane_off : b_lane_off);
    const unsigned a_rd_end = a_rd0 + (unsigned)(kARing * kAStageBytes);
    const unsigned b_rd_end = b_rd0 + (unsigned)(kBRing * kBStageBytes);
    unsigned a_rd = a_rd0, b_rd = b_rd0;

    // Mainloop, compile-time specialized on fast_cta: the fast copy runs
    // predication-free loads; the generic copy keeps full predication.
    // kFastLoop=false instantiates only the generic copy — codegen identical
    // to the pre-peel kernel.
    auto mainloop = [&](auto fastc) {
        constexpr bool kFast = decltype(fastc)::value;
        // Fast-path write carries: one per congruous operand (see
        // PrefetchCarry; crosswise operands get the empty no-op type).
        // Construction targets the first prefetched tile (kStages).
        PrefetchCarry<!kDirectA, T8, kK, kBlockM, kCtaThreads> carry_a(
            a_base, kARing, kAStageBytes, a, a_ld, (int64_t)block_m * kBlockM,
            tid, kStages);
        PrefetchCarry<!kDirectB, T8, kK, kBlockN, kCtaThreads> carry_b(
            b_base, kBRing, kBStageBytes, b, b_ld, (int64_t)block_n * kBlockN,
            tid, kStages);
        // Interleaved prefetch (cuBLAS/CUTLASS loop shape): the next tile's
        // LDGSTS chunks ride inside the MMA phase so their issue slots fill
        // the tensor-pipe gaps ptxas otherwise pads with NOPs (23 NOPs per
        // 32 QMMA here versus 0 in the cuBLAS loop). Full rings only: a lean
        // ring's write slot is the one compute(i) is reading (barrier 4
        // orders the end-of-loop prefetch), so it keeps that placement.
        constexpr bool kInterleave = kFast && !kLeanRing;
        for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        // In the steady state exactly kStages-1 younger groups are in flight
        // when this fires; the tail's unconditional (possibly empty) commits
        // keep that invariant true for every iteration.
        const bool prefetch = tile_index + kStages < tile_count;
        astrai::cp_async_wait_group<kStages - 1>();
        // Barrier 1: every thread's cp.async for this stage is complete
        // before any thread reads tiles written by other threads.
        __syncthreads();

        // Direct chunks for tile i+kStages: issue LDG+PRMT+STS now so the
        // global-load latency hides behind the MMA phase below.
        if (prefetch)
            load_direct(a_stage_of(tile_index + kStages),
                        b_stage_of(tile_index + kStages),
                        (tile_index + kStages) * kK);

        const unsigned a_addr = a_rd;
        const unsigned b_addr = b_rd;
        // Per-k_seg base pair (cuBLAS's scheme): seg s lives at the seg-0
        // base XOR (s<<5) — one LOP3 per extra seg per k-tile, never per
        // fragment. Every LDSM below addresses [base + immediate].
        unsigned a_seg[kSegs], b_seg[kSegs];
#pragma unroll
        for (int s = 0; s < kSegs; ++s) {
            a_seg[s] = a_addr ^ (unsigned)(s * kSegXor);
            b_seg[s] = b_addr ^ (unsigned)(s * kSegXor);
        }

        // kNt ldmatrix.x2 (B) + kMt ldmatrix.x4 (A) feed kMt*kNt*2 mma.sync
        // per k_seg — 0.5 load instructions per MMA, versus 4.5 scalar LDS
        // per MMA in the 128x64-tile version (the kernel was LSU-issue-bound
        // there). B fragments double-buffer across k_segs. kPairB folds the
        // two adjacent nt fragments of one pair into a single x4 (see
        // b4_lane_off above): kNt/2 x4 loads, regs {r0,r1}/{r2,r3} feeding
        // the even/odd nt MMAs respectively.
        unsigned b_frag[2][kNt][2];
        unsigned b_frag4[2][kNt / 2][4];
#pragma unroll
        for (int p = 0; p < kNt / 2; ++p)
            if constexpr (kPairB)
                astrai::ldmatrix_x4_lane(b_frag4[0][p],
                                         b_seg[0] + p * kPairStep);
            else {
                astrai::ldmatrix_x2_lane(b_frag[0][p * 2],
                                         b_seg[0] + p * 2 * kNtStep);
                astrai::ldmatrix_x2_lane(b_frag[0][p * 2 + 1],
                                         b_seg[0] + (p * 2 + 1) * kNtStep);
            }
#pragma unroll
        for (int k_seg = 0; k_seg < kSegs; ++k_seg) {
            const int bcur = k_seg & 1, bnext = bcur ^ 1;
            if (k_seg + 1 < kSegs) {
#pragma unroll
                for (int p = 0; p < kNt / 2; ++p)
                    if constexpr (kPairB)
                        astrai::ldmatrix_x4_lane(
                            b_frag4[bnext][p],
                            b_seg[k_seg + 1] + p * kPairStep);
                    else {
                        astrai::ldmatrix_x2_lane(
                            b_frag[bnext][p * 2],
                            b_seg[k_seg + 1] + p * 2 * kNtStep);
                        astrai::ldmatrix_x2_lane(
                            b_frag[bnext][p * 2 + 1],
                            b_seg[k_seg + 1] + (p * 2 + 1) * kNtStep);
                    }
            }
        // Software-pipelined A fragments: the ldmatrix.x4 for row mt+1
        // is issued before the MMAs consuming row mt, so the LDS fixed
        // latency hides behind tensor-pipe work (cuts the `wait` stall,
        // ~2.3 cycles/issue before this). Costs 4 extra registers.
        // (Cross-k_seg prefetch of row 0 was tried and reverted: the
        // register handoff broke ptxas's software pipelining — 171T → 95T
        // at 2048³; the tensor pipe is issue-bound and the seg-start LDS
        // already hides behind the b-fragment issue order.)
        unsigned a_frag[kMt + 1][4];
        astrai::ldmatrix_x4_lane(a_frag[0], a_seg[k_seg]);
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
            if (mt + 1 < kMt)
                astrai::ldmatrix_x4_lane(a_frag[mt + 1],
                                         a_seg[k_seg] + (mt + 1) * kMtStep);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const unsigned* bops =
                    kPairB ? (b_frag4[bcur][nt >> 1] + (nt & 1) * 2)
                           : b_frag[bcur][nt];
                astrai::mma_sync<T8>(acc[nt][mt], a_frag[mt], bops,
                                     acc[nt][mt]);
            }
        }
        // Next tile's LDGSTS chunks inside the MMA phase: A's after the
        // first k_seg's MMA batch, B's after the last.
        if constexpr (kInterleave) {
            if (k_seg == 0) carry_a.emit(prefetch);
            if (k_seg == kSegs - 1) carry_b.emit(prefetch);
        }
        }
        // Barrier 4 (lean-ring only): every thread finished reading this
        // stage's tiles before the prefetch for the (i+kStages)-th tile
        // overwrites them. With the kStages+1 canonic rotation the prefetch
        // targets the slot compute(i-1) released before barrier 1, so the
        // full-ring path skips this barrier entirely — one __syncthreads per
        // k-tile.
        if constexpr (kLeanRing) __syncthreads();
        if constexpr (!kInterleave) {
            if (prefetch) {
                if constexpr (kFast) {
                    carry_a.emit(true);
                    carry_b.emit(true);
                } else {
                    load_async(a_stage_of(tile_index + kStages),
                               b_stage_of(tile_index + kStages),
                               (tile_index + kStages) * kK);
                }
            }
        }
        // Unconditional commit: empty in the tail, it pads the group
        // sequence so the fixed wait above stays correct (and the
        // predicated-off chunks' zero-fill lands in the slot compute(i-1)
        // released — nothing reads it again before the epilogue drain).
        astrai::cp_async_commit_group();
        // Advance the carries: one stage slot forward, wrapping on the
        // exact ring boundary.
        a_rd += (unsigned)kAStageBytes;
        if (a_rd == a_rd_end) a_rd = a_rd0;
        b_rd += (unsigned)kBStageBytes;
        if (b_rd == b_rd_end) b_rd = b_rd0;
        if constexpr (kFast) {
            carry_a.advance(kAStageBytes);
            carry_b.advance(kBStageBytes);
        }
        }
    };  // mainloop
    // NOTE: a cross-k-tile fragment pipeline (kAheadFrag — head-load the
    // current tile's tail-seg fragments, tail-preload the next tile's
    // seg-0 behind a tightened wait(kStages-2), mirroring cuBLAS's third
    // SASS mechanism) was implemented and measured here: correct, but
    // neutral-to-negative on L20 (-7% at 512³'s sub-wave grid, noise
    // elsewhere). The shallower effective pipeline (kStages-1 groups in
    // flight) costs what the LDSM spreading saves at these tile counts,
    // and ptxas loses its within-iteration software pipelining across the
    // iteration-boundary register handoff. Removed; see
    // perf/fp8_gemm_optimization.md's reverted-experiments table.
    if constexpr (kFastLoop) {
        if (fast_cta)
            mainloop(std::true_type{});
        else
            mainloop(std::false_type{});
    } else {
        mainloop(std::false_type{});
    }
    // Drain the pipeline before the epilogue reclaims the operand rings for
    // output staging: the loop's last commits (possibly only zero-filling
    // predicated-off chunks) are nobody's wait target anymore.
    astrai::cp_async_wait_all();

    // Direct bf16 epilogue through the operand shared memory: the A/B rings
    // are dead once the mainloop ends, so their space stages the output tile
    // (kBlockM x kBlockN bf16, always <= the ring budget). Threads first
    // scatter their accumulators into the tile (STS.32 of bf16x2 pairs), a
    // barrier makes the tile coherent, then the whole CTA copies it out in
    // fully-coalesced 16B chunks. The direct per-thread stores this replaces
    // hit 8 disjoint 16B segments per warp (rows are n*2 bytes apart), ~50%
    // write efficiency — measurable at 2048+ where the epilogue is ~8% of
    // runtime. The 16B-chunk XOR swizzle (chunk index ^ row) keeps both the
    // scatter and the gather conflict-free: a lane quad's chunk and the 8
    // rows of one gather phase map to distinct 4-bank groups.
    const float output_scale = scale;
    // Fused bias (idea B): added to the fp32 accumulator before the single
    // bf16 rounding — one fewer rounding than the out + bias elementwise
    // pass this replaces, and no extra kernel launch / m*n round-trip. The
    // per-lane loads (2 per nt, kMt-times re-read) are L1 broadcasts; rows
    // past the N edge skip the load (their smem slots never copy out).
    const __nv_bfloat16* bias =
        reinterpret_cast<const __nv_bfloat16*>(p.bias_ptr);
    __nv_bfloat16* tile_out = reinterpret_cast<__nv_bfloat16*>(fp8_gemm_smem);
    constexpr int kRowChunks = kBlockN / 8;  // 16B chunks per tile row
    static_assert(kBlockM * kBlockN * 2 <=
                      kARing * kBlockM * kK + kBRing * kBlockN * kK,
                  "output tile must fit the reclaimed operand smem");
    // Swizzled address of one 16B chunk (row r, chunk c) of the tile.
    auto out_chunk = [&](int r, int c) -> __nv_bfloat16* {
        return tile_out + (size_t)r * kBlockN +
               ((c ^ (r & (kRowChunks - 1))) * 8);
    };
    const int local_col0 = warp_n * Traits::kWarpN + thread_in_group * 2;
    const int64_t bias_col0 = (int64_t)block_n * kBlockN;
#pragma unroll
    for (int nt = 0; nt < kNt; ++nt) {
        const int col = local_col0 + nt * 8;
        const int64_t gcol = bias_col0 + col;
        const float b0 =
            bias && gcol < n ? __bfloat162float(bias[gcol]) : 0.0f;
        const float b1 =
            bias && gcol + 1 < n ? __bfloat162float(bias[gcol + 1]) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
            const int r0 = warp_m * Traits::kWarpM + group + mt * 16;
            const float* tile_acc = acc[nt][mt];
            // Two bf16x2 stores per accumulator tile: rows g and g+8 of the
            // m16n8 output, columns tig*2 and tig*2+1 inside one 16B chunk.
            const int off = col & 7;  // element offset within the chunk
            *reinterpret_cast<__nv_bfloat162*>(out_chunk(r0, col >> 3) + off) =
                __floats2bfloat162_rn(tile_acc[0] * output_scale + b0,
                                      tile_acc[1] * output_scale + b1);
            *reinterpret_cast<__nv_bfloat162*>(out_chunk(r0 + 8, col >> 3) +
                                               off) =
                __floats2bfloat162_rn(tile_acc[2] * output_scale + b0,
                                      tile_acc[3] * output_scale + b1);
        }
    }
    __syncthreads();
    // Coalesced copy-out: thread -> one 16B chunk; consecutive threads walk
    // a row so each global transaction covers a full 128B line.
    const int64_t row0_global = (int64_t)block_m * kBlockM;
    const int64_t col0_global = (int64_t)block_n * kBlockN;
    constexpr int kTotalChunks = kBlockM * kRowChunks;
    for (int idx = tid; idx < kTotalChunks; idx += kCtaThreads) {
        const int r = idx / kRowChunks;
        const int c = idx % kRowChunks;
        const int64_t row = row0_global + r;
        if (row >= m) break;  // rows are consecutive: nothing left in range
        const int64_t col = col0_global + (int64_t)c * 8;
        const uint4 v = *reinterpret_cast<const uint4*>(out_chunk(r, c));
        auto* dst = out_bf16 + row * n + col;
        if (col + 8 <= n && (reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
            if constexpr (kStreamOut) {
                // Evict-first streaming store knob. Measured neutral on
                // L20 squares and -3..4% on rects (the evict-first policy
                // hurts more than the L2 B-tile protection helps at these
                // sizes); kept as a template knob for other SKUs. Default
                // off.
                __stcs(reinterpret_cast<uint4*>(dst), v);
            } else {
                *reinterpret_cast<uint4*>(dst) = v;
            }
        } else {
            // N-tail chunk or an odd-n row base: spill the elements that
            // survive the row edge (and stay aligned).
            const __nv_bfloat16* elems =
                reinterpret_cast<const __nv_bfloat16*>(&v);
            for (int e = 0; e < 8 && col + e < n; ++e) dst[e] = elems[e];
        }
    }
}

// ---------------------------------------------------------------------------
// Launchers — pure CUDA (no torch), usable from the binding and pure C tests.
// ---------------------------------------------------------------------------

// SM count of the current device (cached per device; benign init race —
// every writer stores the same value). Host-side only: feeds the
// device-adaptive dispatch thresholds.
inline int device_sm_count() {
    static int cached[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        return sms > 0 ? sms : 1;
    }
    if (!cached[dev]) {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cached[dev] = sms > 0 ? sms : 1;
    }
    return cached[dev];
}


// Launch one kernel instantiation with its shared-memory budget: stages live
// in dynamic smem, so budgets beyond the 48KB static limit opt in once per
// instantiation via cudaFuncSetAttribute (see AGENTS.md "dynamic shared
// memory"). Templated on the kernel *value* (auto NTTP) so every
// instantiation owns its own armed flag — same-signature kernels must not
// share it (the attribute is per-function).
template <auto Kernel, typename... Args>
void launch_with_smem(int smem_bytes, dim3 grid, dim3 block,
                      cudaStream_t stream, Args... args) {
    if (smem_bytes > 48 * 1024) {
        static bool armed = false;  // per instantiation
        if (!armed) {
            cudaFuncSetAttribute(Kernel, 
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes);
            armed = true;
        }
    }
    Kernel<<<grid, block, smem_bytes, stream>>>(args...);
}

// Pre-quantized GEMM tile config: 128x128 CTA (8 warps x 64x32 warp tiles).
// kK selects the K tile (32 / 64 / 128; larger kK halves the __syncthreads
// count per K and doubles the MMA work per stage at more smem per stage).
// Stages is the cp.async pipeline depth (smem = Stages * (BM + BN) * kK
// bytes for congruous layouts; deep pipelines are dynamic-smem backed, 1
// CTA/SM past 48KB). GroupRaster defaults to the historically-measured best
// per LayoutA (grouped for A-crosswise, plain for A-congruous).
// Crosswise operands always take load_crosswise_direct — the alternative
// staging+transpose pipeline measured 15-20% slower everywhere probed
// (contract k 2048..32768, DRAM-streaming B included) and was removed.

// Shape-based tile dispatch (grid-searched on the production shapes, see
// perf/fp8_sweep.cu): small outputs take 64x64 CTAs of 32x32 warps
// with a lean (kStages-deep) ring: 24KB of smem keeps 4 CTAs resident, and
// the extra blocks fill the wave quantization gap (512^3: 64 vs 16 CTAs).
// The large-output path takes the 128x128 CTA (8 warps x 64x32) with the
// kStages+1 ring — one __syncthreads per k-tile and ~200 TF at scale.
// The threshold applies to the TOTAL tile count (batch x per-matrix tiles):
// batched runs keep full per-matrix CTA efficiency once the aggregate grid
// saturates the device (measured 64x512^3: big 160 vs small 123 TF — a
// per-matrix-only threshold lost 30%). m <= 64 always takes the small CTA:
// a 128-row CTA would waste half its MMA work on predicated-off rows.
//
// Wave-quantization makes the crossover non-monotonic (92-SM L20, cubes,
// congruous NT): the 128x128 CTA wins inside one full wave (81 tiles: big
// +24%) and from ~1.5 waves up (144: +23%, 256: +39%, 2048^3 123->171 TF),
// but loses inside the quantization dip just past one wave (100 tiles =
// 1.09 waves) — with the OLD loop, whose big CTA stalled on the tail wave's
// exposed pipeline drain. The interleaved-prefetch loop (perf 6.2) removed
// that stall, and the trade flipped across the whole measured band: 1280^3
// (1.09 waves) big 163T vs small 100T, 4096x512x4096 (1.4 waves) big 139T
// vs small 113T. The small CTA now only serves the genuinely sub-wave band
// below 5/8 of a wave (and m <= 64, where a 128-row CTA wastes half its
// rows); inside [5/8, 1] waves the big CTA was already the measured winner
// (63-tile rect +8%).
inline bool prefer_small_cta(int64_t tiles_128, int64_t m) {
    if (m <= 64) return true;
    return tiles_128 < device_sm_count() * 5 / 8;
}

template <FP8Format Fmt, typename LayoutA = RowMajor,
          typename LayoutB = RowMajor, int kK = 64, int Stages = 2,
          int GroupRaster = (std::is_same_v<LayoutA, ColMajor> ||
                             std::is_same_v<LayoutB, ColMajor>)
                                ? 8
                                : 0>
void launch_fp8_gemm(const FP8Params& p, cudaStream_t stream) {
    // m <= 64 and small total outputs share the 64x64 small CTA (with the
    // predication-free interior loop); the predicate counts batch x
    // per-matrix tiles (see prefer_small_cta).
    const int64_t tiles_128 =
        (int64_t)p.batch * ((p.m + 127) / 128) * ((p.n + 127) / 128);
    if (prefer_small_cta(tiles_128, p.m)) {
        dim3 grid((p.n + 63) / 64, (p.m + 63) / 64, p.batch);
        // Full-ring small CTAs — ONE __syncthreads per k-tile, cuBLAS's
        // barrier structure (the lean ring traded a second barrier for a
        // 4th resident CTA and measured slower: the barrier costs more
        // than the residency buys, e.g. 1280³ +5..9%). Two depths by grid
        // shape: the 24KB 3-slot s2 variant keeps 4 CTAs/SM while the
        // whole grid stays resident (<= one 3-CTA wave); past that the
        // 32KB s3 variant's deeper cp.async pipeline wins on multi-wave
        // grids (measured 1280³: 107T vs 98T; sub-wave grids tie within
        // +-1%). kFastLoop stays on: the predication-free interior load
        // is where the small CTA's issue budget goes.
        const int64_t tiles_64 =
            (int64_t)p.batch * ((p.m + 63) / 64) * ((p.n + 63) / 64);
        if (tiles_64 <= (int64_t)device_sm_count() * 3) {
            using Traits = Fp8GemmTraits<Fmt, 64, 64, kK, 2, 32, 32>;
            launch_with_smem<
                fp8_gemm_kernel<Traits, LayoutA, LayoutB, GroupRaster, false,
                                false, true>>(
                Fp8GemmSmem<Traits, LayoutA, LayoutB, false>::kBytes, grid,
                dim3(Traits::kCtaThreads), stream, p);
        } else {
            using Traits = Fp8GemmTraits<Fmt, 64, 64, kK, 3, 32, 32>;
            launch_with_smem<
                fp8_gemm_kernel<Traits, LayoutA, LayoutB, GroupRaster, false,
                                false, true>>(
                Fp8GemmSmem<Traits, LayoutA, LayoutB, false>::kBytes, grid,
                dim3(Traits::kCtaThreads), stream, p);
        }
        return;
    }
    using Traits = Fp8GemmTraits<Fmt, 128, 128, kK, Stages>;
    dim3 grid((p.n + 127) / 128, (p.m + 127) / 128, p.batch);
    // Interior-loop specialization on the big CTA as well: with the base-pair
    // fragment addressing the doubled mainloop no longer spills, and the
    // predication-free loads win across the band (measured, L20: 1024^3
    // 98->103T, 2048^3 172->177T, 8192^3 201->205T, 896x1280 124->135T; the
    // pre-base-pair attempt regressed ~3% at 131 regs). Only congruous
    // layouts can enter fast_cta, so crosswise (TN) instantiations keep the
    // single generic body — no dead second loop in their I-cache.
    constexpr bool kBigFast = !std::is_same_v<LayoutA, ColMajor> &&
                              !std::is_same_v<LayoutB, RowMajor>;
    launch_with_smem<
        fp8_gemm_kernel<Traits, LayoutA, LayoutB, GroupRaster, false, false,
                        kBigFast>>(
        Fp8GemmSmem<Traits, LayoutA, LayoutB, false>::kBytes, grid,
        dim3(Traits::kCtaThreads), stream, p);
}

}  // namespace fp8
}  // namespace astrai
