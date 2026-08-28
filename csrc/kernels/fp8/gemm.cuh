#pragma once
// FP8 GEMM device code — pure CUDA, no torch. Kernels take the FP8Params
// POD; tile shape, formats and layout tags ride on one Policy template
// parameter (CUTLASS-style), and launchers are plain functions shared by
// the torch binding and the C tests.

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

// log2 of a compile-time power of two (for the swizzle shifts).
template <int N, int Acc = 0>
struct log2_const : log2_const<(N >> 1), Acc + 1> {};
template <int Acc>
struct log2_const<1, Acc> {
    static constexpr int value = Acc;
};

// ---------------------------------------------------------------------------
// Shared device helpers
// ---------------------------------------------------------------------------
// The FP8 MMA lives in astrai::mma_sync (common/mma.cuh), instantiated with
// the kernel's T8 and accumulating in-place. The cp.async primitives live
// in common/cp_async.cuh.

// Swizzled address inside a flat [rows * K] staging tile: the 16B chunk
// index is XORed with the row bits at [3, 3+log2(kChunks)) so a warp's
// ldmatrix fragment load (8 consecutive rows x 16B) hits all 32 banks
// exactly once; chunks stay contiguous, so cp.async staging is unaffected.
template <int K, typename T8>
__device__ __forceinline__ T8* tile_at(T8* tile, int row, int col) {
    constexpr int kChunks = K / 16;  // 16B chunks per row
    static_assert(kChunks >= 1 && (kChunks & (kChunks - 1)) == 0,
                  "swizzle needs a power-of-two 16B-chunk count");
    constexpr int kShift = 3 - log2_const<kChunks>::value;
    return tile + row * K +
           ((((col >> 4) ^ ((row >> kShift) & (kChunks - 1))) << 4) + (col & 15));
}

// Stage-load a CONGRUOUS operand (contract-contiguous storage — the only
// cp.async-able shape) into the flat [rows * K] swizzled tile. kInterior
// drops all predication: valid only for a fully interior CTA (whole rows,
// 16B-aligned base|ld, k_base + K <= contract — the fast_cta peel
// guarantees these); a thread's chunk run is swizzle-invariant
// ((n+j)^swz == (n^swz)^j), so the address math folds to one immediate XOR
// per chunk. Crosswise operands go through load_crosswise_direct instead.
template <typename T8, int K, int RowsTile, int kThreads,
          bool kInterior = false>
__device__ __forceinline__ void
load_operand_tile(T8* tile, const T8* __restrict__ operand, int64_t rows,
                  int64_t contract, int64_t ld, int tid, int64_t k_base,
                  int64_t block_row) {
    constexpr int kChunks = K / 16;
    static_assert(RowsTile * kChunks % kThreads == 0,
                  "tile chunks must divide evenly across threads");
    constexpr int kCpt = RowsTile * kChunks / kThreads;  // chunks per thread
    constexpr int kCpr = kChunks / kCpt;  // chunks per row slice
    const int r = tid / kCpr;
    const int c0 = (tid % kCpr) * kCpt * 16;
    if constexpr (kInterior) {
        const char* src = reinterpret_cast<const char*>(
            operand + (block_row + r) * ld + k_base + c0);
        const uintptr_t dst =
            reinterpret_cast<uintptr_t>(tile_at<K>(tile, r, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(reinterpret_cast<T8*>(dst ^ (j << 4)),
                                src + j * 16);
    } else {
        const int64_t row = block_row + r;
        const bool row_ok = row < rows;
        // k_base and every c are multiples of 16, so all chunks share the
        // row base's alignment verdict.
        const auto* src = operand + row * ld + k_base;
        const bool chunk_aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
#pragma unroll
        for (int j = 0; j < kCpt; ++j) {
            const int c = c0 + j * 16;
            T8* dst = tile_at<K>(tile, r, c);
            if (row_ok && chunk_aligned && k_base + c + 15 < contract) {
                astrai::cp_async_16(dst, src + c);
            } else {
                // Tail chunk / misaligned base / OOB row: scalar fill.
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    dst[i] =
                        row_ok && k_base + c + i < contract ? src[c + i] : T8(0.0f);
            }
        }
    }
}

// Loop-carried prefetch state for one congruous operand ring: per-thread
// (r, c0) mapping with the swizzled stage destination and global source
// pointer carried across k-tiles, so each prefetch chunk is one LDGSTS
// issued straight from registers. The guard is a property of the operand's
// layout, so it lives in the type: the false specialization (crosswise
// operand) is an empty no-op — no dead declarations, no if constexpr at
// the use sites.
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

    // Emit this thread's chunks for the current tile; pf false (loop tail)
    // zero-fills into the slot compute(i-1) already released.
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
// Pre-quantized GEMM kernel: FP8 A/B staged into shared memory, FP32
// accumulation, BF16 output. Operands materialize in the compact canonical
// [rows][kK] tile so MMA fragments read directly — no in-kernel transpose.
// ---------------------------------------------------------------------------

// Direct (synchronous) crosswise load into a canonical rotating stage:
// LDG.128 x4 (4 consecutive contract bytes x 16 rows) + in-register PRMT
// transpose + 16 STS.32. Crosswise operands cannot cp.async into the
// canonical tile (a 16B global run holds one contract byte for each of 16
// rows), so they take this path.
template <typename T8, int K, int RowsTile, int kThreads>
__device__ __forceinline__ void
load_crosswise_direct(T8* tile, const T8* __restrict__ operand, int64_t rows,
                      int64_t contract, int64_t ld, int tid, int64_t k_base,
                      int64_t block_row) {
    constexpr int kQuads = K / 4;    // 4-byte contract quads per tile
    constexpr int kGroups = RowsTile / 16;
    constexpr int kTChunks = kQuads * kGroups;  // 64B chunks per tile
    // r0 is a multiple of 16 and p*ld preserves alignment whenever ld has
    // it, so every run of a chunk shares one alignment verdict.
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
                // [v0.b(i), v1.b(i), v2.b(i), v3.b(i)].
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

// Layout-aware shared-memory budget and occupancy hint. Every operand ring
// holds kStages+1 buffers: the load for tile i+kStages targets slot
// (i-1)%(kStages+1) — already consumed — so neither load path needs a
// post-compute barrier (one __syncthreads per k-tile). The 48KB static
// watermark picks the resident-CTA hint for __launch_bounds__.
template <typename Traits, typename LayoutA, typename LayoutB>
struct Fp8GemmSmem {
    // Crosswise (direct-load) operands: A ColMajor storage, B RowMajor
    // storage (B's tag is relative to the canonical [K][N]).
    static constexpr bool kDirectA = std::is_same_v<LayoutA, ColMajor>;
    static constexpr bool kDirectB = std::is_same_v<LayoutB, RowMajor>;
    static constexpr int kRingDepth = Traits::kStages + 1;
    static constexpr int kBytes =
        kRingDepth * (Traits::kBlockM + Traits::kBlockN) * Traits::kK;
    static constexpr int kMinCtas = kBytes <= 48 * 1024 ? 2 : 1;
};

// ---------------------------------------------------------------------------
// Kernel policy: one type per kernel instantiation (CUTLASS-style
// consolidation) — traits + layout tags + scheduling knobs, the single
// template parameter the kernel and both collectives take.
template <FP8Format Fmt_, int BlockM_, int BlockN_, typename LayoutA_,
          typename LayoutB_, int WarpM_, int WarpN_, int kK_, int Stages_,
          int GroupRaster_, bool StreamOut_ = false, bool FastLoop_ = false>
struct Fp8GemmPolicy {
    using Traits =
        Fp8GemmTraits<Fmt_, BlockM_, BlockN_, kK_, Stages_, WarpM_, WarpN_>;
    using LayoutTagA = LayoutA_;
    using LayoutTagB = LayoutB_;
    static constexpr int kGroupRaster = GroupRaster_;
    static constexpr bool kStreamOut = StreamOut_;
    static constexpr bool kFastLoop = FastLoop_;
    using Smem = Fp8GemmSmem<Traits, LayoutA_, LayoutB_>;
    // Flattened for __launch_bounds__, which takes no dependent type names.
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr int kMinCtas = Smem::kMinCtas;
    static constexpr int kSmemBytes = Smem::kBytes;
};

// LayoutA / LayoutB tag the operands' storage; the kernel always computes
// out[m][n] = sum_p tileA[m][p] * tileB[n][p] with tiles materialized in
// the canonical [M][kK] / [N][kK] layout, so the tags only change how the
// stage-load gathers from global memory. With p.out_transposed set (the
// swap dispatch for NN problems) the kernel runs the transposed problem
// E = B^T * A^T and the epilogue scatters D[m][n] = E[n][m]; bias then
// indexes D-cols, i.e. the kernel's rows.
//
// The kernel decomposes CUTLASS-style into three collectives:
//   Fp8GemmTileScheduler   — CTA id -> (block_m, block_n) raster order
//   Fp8CollectiveMainloop  — stage rings, gmem->smem loads, mma.sync loop
//   Fp8CollectiveEpilogue  — fused bias + bf16 scatter + coalesced copy-out
// with fp8_gemm_kernel as the thin orchestrator.

// ---------------------------------------------------------------------------
// Tile scheduler: the linear CTA id maps to (block_m, block_n) in grouped
// (L2-friendly) raster — consecutive CTAs share one B column stripe — or
// plain N-fastest raster (kRasterGroup=0, the measured best for dX's
// crosswise-B layouts where grouping was neutral).
template <int kRasterGroup>
struct Fp8GemmTileScheduler {
    static __device__ int2 tile(const uint3& block, const dim3& blocks) {
        if constexpr (kRasterGroup > 0) {
            constexpr int kGroupM = kRasterGroup;
            const int bid = int(block.y) * int(blocks.x) + int(block.x);
            const int group_first_m = (bid / (kGroupM * int(blocks.x))) * kGroupM;
            const int group_rows =
                min(int(blocks.y) - group_first_m, kGroupM);  // M-tail group is short
            return int2{group_first_m + bid % group_rows,
                        (bid % (kGroupM * int(blocks.x))) / group_rows};
        } else {
            return int2{int(block.y), int(block.x)};
        }
    }
};

// ---------------------------------------------------------------------------
// Collective mainloop: shared-memory stage rings, the gmem->smem stage loads
// (congruous cp.async / crosswise LDG+PRMT), the per-lane ldmatrix fragment
// addressing and the software-pipelined mma.sync loop.
template <typename Policy>
struct Fp8CollectiveMainloop {
    using Traits = typename Policy::Traits;
    using LayoutA = typename Policy::LayoutTagA;
    using LayoutB = typename Policy::LayoutTagB;
    using Smem = Fp8GemmSmem<Traits, LayoutA, LayoutB>;
    static constexpr bool kFastLoop = Policy::kFastLoop;
    using T8 = std::conditional_t<Traits::kIsE5M2, __nv_fp8_e5m2, __nv_fp8_e4m3>;
    static constexpr int kBlockM = Traits::kBlockM;
    static constexpr int kBlockN = Traits::kBlockN;
    static constexpr int kK = Traits::kK;
    static constexpr int kStages = Traits::kStages;
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr bool kDirectA = Smem::kDirectA;
    static constexpr bool kDirectB = Smem::kDirectB;
    static_assert(kStages >= 1 && kStages <= 8,
                  "FP8 GEMM stages must be in [1, 8]");
    // CTA = (BlockM/WarpM) x (BlockN/WarpN) warps, each warp computing
    // kMt x kNt m16n8k32 MMAs. Rings rotate kStages+1 buffers (see
    // Fp8GemmSmem) — one __syncthreads per k-tile.
    static constexpr int kMt = Traits::kWarpM / 16;  // 16-row MMA tiles per warp
    static constexpr int kNt = Traits::kWarpN / 8;   // 8-col MMA tiles per warp
    static constexpr int kSegs = kK / kMmaK;  // mma-sized k segments per tile
    static constexpr int kARing = Smem::kRingDepth;
    static constexpr int kBRing = Smem::kRingDepth;
    static constexpr int kAStageBytes = kBlockM * kK;
    static constexpr int kBStageBytes = kBlockN * kK;

    T8* const a_base;
    T8* const b_base;
    const T8* const a;
    const T8* const b;
    const int64_t m, n, k, a_ld, b_ld;
    const int tid;
    const int64_t block_m, block_n;
    const int warp_m, warp_n;
    const int a_row0;  // + mt * 16 in the loop
    const int b_row0;  // + nt * 8
    const int64_t tile_count;
    // Interior-CTA peel (kFastLoop instantiations only): whole-CTA,
    // 16B-aligned, K without tail — the mainloop then runs a compile-time
    // specialized copy with no per-chunk predication (measured +4.5..10% on
    // the issue-bound small CTA; the 128x128 CTA regressed, so only the
    // small CTA opts in). The verdict is uniform per CTA.
    const bool fast_cta;

    __device__ Fp8CollectiveMainloop(char* smem, const T8* a, const T8* b,
                                     int64_t m, int64_t n, int64_t k,
                                     int64_t a_ld, int64_t b_ld, int tid,
                                     int2 block)
        : a_base(reinterpret_cast<T8*>(smem)),
          b_base(reinterpret_cast<T8*>(smem + kARing * kAStageBytes)),
          a(a), b(b), m(m), n(n), k(k), a_ld(a_ld), b_ld(b_ld), tid(tid),
          block_m(block.x), block_n(block.y),
          warp_m((tid >> 5) / Traits::kWarpsN),
          warp_n((tid >> 5) % Traits::kWarpsN),
          a_row0(warp_m * Traits::kWarpM),
          b_row0(warp_n * Traits::kWarpN),
          tile_count((k + kK - 1) / kK),
          fast_cta(kFastLoop && !kDirectA && !kDirectB &&
                   ((int64_t)block.x * kBlockM + kBlockM <= m) &&
                   ((int64_t)block.y * kBlockN + kBlockN <= n) &&
                   ((reinterpret_cast<uintptr_t>(a) | (uint64_t)a_ld) & 15) == 0 &&
                   ((reinterpret_cast<uintptr_t>(b) | (uint64_t)b_ld) & 15) == 0 &&
                   (k % kK) == 0) {}

    // Stage-slot helpers: rings rotate one slot per k-tile, so callers
    // either compute the slot from the tile index (prologue, generic loop)
    // or carry an advancing pointer (steady-state fast loop).
    __device__ __forceinline__ T8* a_stage_of(int64_t tile) const {
        return a_base + (size_t)(tile % kARing) * kAStageBytes;
    }
    __device__ __forceinline__ T8* b_stage_of(int64_t tile) const {
        return b_base + (size_t)(tile % kBRing) * kBStageBytes;
    }
    // Asynchronous congruous loads for one k-tile: cp.async into the
    // canonical rings; kFast selects the predication-free interior copy
    // (fast_cta admits only congruous operands). Called after the
    // post-compute barrier, alongside the commit.
    template <bool kFast = false>
    __device__ __forceinline__ void load_async(T8* a_stage, T8* b_stage,
                                               int64_t k_base) const {
        if constexpr (!kDirectA)
            load_operand_tile<T8, kK, kBlockM, kCtaThreads, kFast>(
                a_stage, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
        if constexpr (!kDirectB)
            load_operand_tile<T8, kK, kBlockN, kCtaThreads, kFast>(
                b_stage, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
    }
    // Synchronous direct-crosswise loads for one k-tile. In the steady
    // state this runs right after barrier 1, so the LDG latency and the
    // PRMT transpose overlap the MMA phase instead of stalling the
    // inter-barrier window (which dominated the dX/dW stall profile).
    __device__ __forceinline__ void load_direct(T8* a_stage, T8* b_stage,
                                                int64_t k_base) const {
        if constexpr (kDirectA)
            load_crosswise_direct<T8, kK, kBlockM, kCtaThreads>(
                a_stage, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
        if constexpr (kDirectB)
            load_crosswise_direct<T8, kK, kBlockN, kCtaThreads>(
                b_stage, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
    }

    // Prime the pipeline: kStages committed groups, one per stage slot.
    // The commit is unconditional — when K is shorter than the pipeline the
    // skipped stages commit empty groups, so the group sequence stays
    // tile-indexed and the steady-state wait count never needs a runtime
    // dispatch.
    __device__ __forceinline__ void prologue() const {
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
            if (stage < tile_count) {
                if (fast_cta)
                    load_async<true>(a_stage_of(stage), b_stage_of(stage),
                                     (int64_t)stage * kK);
                else
                    load_async(a_stage_of(stage), b_stage_of(stage),
                               (int64_t)stage * kK);
                load_direct(a_stage_of(stage), b_stage_of(stage),
                            (int64_t)stage * kK);
            }
            astrai::cp_async_commit_group();
        }
    }

    // Steady-state mainloop, compile-time specialized on kFast: the fast
    // copy runs predication-free loads with loop-carried read/write
    // pointers; the generic copy keeps full predication. kFastLoop=false
    // instantiates only the generic copy.
    template <bool kFast>
    __device__ __forceinline__ void run_loop(float acc[kNt][kMt][4]) const {
        const int lane = tid & 31;
        // Fast-path write carries: one per congruous operand (crosswise
        // operands get the empty no-op type), targeting the first
        // prefetched tile (kStages).
        PrefetchCarry<!kDirectA, T8, kK, kBlockM, kCtaThreads> carry_a(
            a_base, kARing, kAStageBytes, a, a_ld, block_m * kBlockM, tid,
            kStages);
        PrefetchCarry<!kDirectB, T8, kK, kBlockN, kCtaThreads> carry_b(
            b_base, kBRing, kBStageBytes, b, b_ld, block_n * kBlockN, tid,
            kStages);
        // Steady-state read carries: the LDSM base of the current k-tile's
        // stage with the lane offset folded in, advanced one stage per
        // iteration with an equality wrap — replaces the per-k-tile
        // (tile % ring) * stage_bytes recomputation (a UIMAD.WIDE
        // magic-division ladder in SASS).
        const unsigned a_rd0 = __cvta_generic_to_shared(a_base) + a_lane_off(lane);
        const unsigned b_rd0 =
            __cvta_generic_to_shared(b_base) +
            (kPairB ? b4_lane_off(lane) : b_lane_off(lane));
        const unsigned a_rd_end = a_rd0 + (unsigned)(kARing * kAStageBytes);
        const unsigned b_rd_end = b_rd0 + (unsigned)(kBRing * kBStageBytes);
        unsigned a_rd = a_rd0, b_rd = b_rd0;
        for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        // In the steady state exactly kStages-1 younger groups are in flight
        // when this fires; the tail's unconditional (possibly empty)
        // commits keep that invariant true for every iteration.
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
        // per k_seg — 0.5 load instructions per MMA. B fragments
        // double-buffer across k_segs; kPairB folds the two adjacent nt
        // fragments of one pair into a single x4 (see b4_lane_off).
        unsigned b_frag[2][kNt][2];
        unsigned b_frag4[2][kNt / 2][4];
        load_b_frags(b_frag[0][0], b_frag4[0][0], b_seg[0]);
#pragma unroll
        for (int k_seg = 0; k_seg < kSegs; ++k_seg) {
            const int bcur = k_seg & 1, bnext = bcur ^ 1;
            if (k_seg + 1 < kSegs)
                load_b_frags(b_frag[bnext][0], b_frag4[bnext][0],
                             b_seg[k_seg + 1]);
        // Software-pipelined A fragments: the ldmatrix.x4 for row mt+1 is
        // issued before the MMAs consuming row mt, so the LDS latency hides
        // behind tensor-pipe work. Costs 4 extra registers.
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
        if constexpr (kFast) {
            if (k_seg == 0) carry_a.emit(prefetch);
            if (k_seg == kSegs - 1) carry_b.emit(prefetch);
        }
        }
        // Generic loop (no interleaved prefetch): the next tile's predicated
        // loads run after the MMA phase.
        if constexpr (!kFast) {
            if (prefetch) {
                load_async(a_stage_of(tile_index + kStages),
                           b_stage_of(tile_index + kStages),
                           (tile_index + kStages) * kK);
            }
        }
        // Unconditional commit: empty in the tail, it pads the group
        // sequence so the fixed wait above stays correct.
        astrai::cp_async_commit_group();
        a_rd += (unsigned)kAStageBytes;
        if (a_rd == a_rd_end) a_rd = a_rd0;
        b_rd += (unsigned)kBStageBytes;
        if (b_rd == b_rd_end) b_rd = b_rd0;
        if constexpr (kFast) {
            carry_a.advance(kAStageBytes);
            carry_b.advance(kBStageBytes);
        }
        }
    }

    __device__ __forceinline__ void accumulate(float acc[kNt][kMt][4]) const {
        if constexpr (kFastLoop) {
            if (fast_cta)
                run_loop<true>(acc);
            else
                run_loop<false>(acc);
        } else {
            run_loop<false>(acc);
        }
    }

  private:
    // Per-lane ldmatrix fragment addressing (base-pair scheme, mirrored
    // from the cuBLAS SASS): one base register per operand per k_seg,
    // every fragment offset an LDSM immediate — zero address arithmetic
    // inside the MMA phase. The XOR swizzle's source bits come only from
    // the lane's row-within-matrix (r7), so the 8/16-row fragment steps
    // never reach them and
    //   addr(s, mt) = lane_base + mt*(16*kK)  ^ (s<<5)     [A, x4]
    //   addr(s, nt) = lane_base + nt*(8*kK)   ^ (s<<5)     [B, x2 / x4]
    __device__ __forceinline__ unsigned a_lane_off(int lane) const {
        const int r7 = lane & 7;          // row within the 8-row matrix
        const int rh8 = (lane >> 3) & 1;  // +8 rows (A: lanes 8-15, 24-31)
        const int rh16 = lane >> 4;       // +1 chunk (A: lanes 16-31)
        constexpr int kChunks = kK / 16;
        constexpr int kShift = 3 - log2_const<kChunks>::value;  // tile_at's shift
        const unsigned lswz =
            static_cast<unsigned>((r7 >> kShift) & (kChunks - 1));
        // Stage-relative, loop-invariant per-lane base; A's fragment row
        // carries the +8-row (rh8) and +1-chunk (rh16) halves.
        return static_cast<unsigned>((a_row0 + rh8 * 8 + r7) * kK +
                                     ((rh16 ^ lswz) << 4));
    }
    __device__ __forceinline__ unsigned b_lane_off(int lane) const {
        const int r7 = lane & 7;
        const int rh8 = (lane >> 3) & 1;  // +8 rows (B uses rh8 as its chunk half)
        constexpr int kChunks = kK / 16;
        constexpr int kShift = 3 - log2_const<kChunks>::value;
        const unsigned lswz =
            static_cast<unsigned>((r7 >> kShift) & (kChunks - 1));
        return static_cast<unsigned>((b_row0 + r7) * kK + ((rh8 ^ lswz) << 4));
    }
    // x4-paired B loads: one ldmatrix.x4 feeds the two adjacent nt
    // fragments. Lane contract: lanes 0-7 address rows n0..n7 chunk c,
    // lanes 8-15 rows n0..n7 chunk c+1, lanes 16-23 rows n8..n15 chunk c,
    // lanes 24-31 rows n8..n15 chunk c+1. The +8-row step never reaches
    // the swizzle source bits for kK <= 64; kK=128 swizzles on row[2:0]
    // where +8 flips bits, so that config keeps the x2 loads.
    static constexpr unsigned kMtStep = 16 * kK;  // bytes per m-tile row step
    static constexpr unsigned kNtStep = 8 * kK;   // bytes per n-tile row step
    static constexpr unsigned kSegXor = 32;       // chunk-index +2 per k_seg
    static constexpr bool kPairB = kK / 16 <= 4;
    static_assert(!kPairB || kNt % 2 == 0, "B pairing needs even kNt");
    static constexpr unsigned kPairStep = 16 * kK;  // bytes per nt-pair row step
    __device__ __forceinline__ unsigned b4_lane_off(int lane) const {
        return b_lane_off(lane) + (lane >> 4) * kPairStep / 2;
    }

    // One k_seg's B-fragment loads, shared by the initial fill and the
    // double-buffer's next-seg fill. frag2/frag4 are the flat bases of one
    // b_frag / b_frag4 buffer (the unused one is never touched).
    __device__ __forceinline__ void
    load_b_frags(unsigned* frag2, unsigned* frag4, unsigned seg_base) const {
#pragma unroll
        for (int p = 0; p < kNt / 2; ++p) {
            if constexpr (kPairB) {
                astrai::ldmatrix_x4_lane(frag4 + p * 4,
                                         seg_base + p * kPairStep);
            } else {
                astrai::ldmatrix_x2_lane(frag2 + p * 4,
                                         seg_base + p * 2 * kNtStep);
                astrai::ldmatrix_x2_lane(frag2 + p * 4 + 2,
                                         seg_base + (p * 2 + 1) * kNtStep);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Collective epilogue: fused bias, the bf16 scatter of the fp32 accumulators
// through the reclaimed operand shared memory, and the coalesced copy-out.
template <typename Policy>
struct Fp8CollectiveEpilogue {
    using Traits = typename Policy::Traits;
    static constexpr bool kStreamOut = Policy::kStreamOut;
    static constexpr int kBlockM = Traits::kBlockM;
    static constexpr int kBlockN = Traits::kBlockN;
    static constexpr int kMt = Traits::kWarpM / 16;
    static constexpr int kNt = Traits::kWarpN / 8;

    __nv_bfloat16* const tile_out;
    const float output_scale;
    const __nv_bfloat16* const bias;
    const int64_t m, n;
    const bool t_out;
    const int row_elems, row_chunks;
    const int warp_m, warp_n, group, thread_in_group;
    const int64_t block_m, block_n;

    __device__ Fp8CollectiveEpilogue(char* smem, const FP8Params& p,
                                     int64_t block_m, int64_t block_n, int tid)
        : tile_out(reinterpret_cast<__nv_bfloat16*>(smem)),
          output_scale(*p.scale),
          bias(reinterpret_cast<const __nv_bfloat16*>(p.bias_ptr)),
          m(p.m), n(p.n), t_out(p.out_transposed != 0),
          row_elems(t_out ? kBlockM : kBlockN),
          row_chunks(row_elems / 8),
          warp_m((tid >> 5) / Traits::kWarpsN),
          warp_n((tid >> 5) % Traits::kWarpsN),
          group((tid & 31) >> 2),
          thread_in_group(tid & 3),
          block_m(block_m), block_n(block_n) {}

    // Swizzled address of one 16B chunk (row r, chunk c) of the staged
    // tile. Plain orientation: kBlockM rows of kBlockN elems; out-
    // transposed (swap dispatch): rows and row length trade places. Both
    // row-chunk counts are powers of two, keeping the XOR swizzle
    // well-defined.
    __device__ __forceinline__ __nv_bfloat16* out_chunk(int r, int c) const {
        return tile_out + (size_t)r * row_elems +
               ((c ^ (r & (row_chunks - 1))) * 8);
    }
    __device__ __forceinline__ __nv_bfloat16* out_elem(int r, int v) const {
        return out_chunk(r, v >> 3) + (v & 7);
    }

    // Scatter the accumulators into the staging tile: the operand rings are
    // dead once the mainloop ends, so their space stages the bf16 output
    // tile. Threads scatter (STS.32 of bf16x2 pairs), a barrier makes the
    // tile coherent, then the whole CTA copies it out in fully-coalesced
    // 16B chunks. The 16B-chunk XOR swizzle keeps both the scatter and the
    // gather conflict-free.
    __device__ __forceinline__ void stage(float acc[kNt][kMt][4]) const {
        // Fused bias: added to the fp32 accumulator before the single bf16
        // rounding. The per-lane loads are L1 broadcasts; rows past the
        // edge skip the load (their smem slots never copy out). Under
        // out_transposed the bias indexes D-cols = the kernel's rows.
        const int local_col0 = warp_n * Traits::kWarpN + thread_in_group * 2;
        const int64_t bias_col0 = block_n * kBlockN;
        const int64_t bias_row0 = block_m * kBlockM;
        if (!t_out) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = local_col0 + nt * 8;
                const int64_t gcol = bias_col0 + col;
                const float b0 =
                    bias && gcol < n ? __bfloat162float(bias[gcol]) : 0.0f;
                const float b1 =
                    bias && gcol + 1 < n ? __bfloat162float(bias[gcol + 1])
                                         : 0.0f;
#pragma unroll
                for (int mt = 0; mt < kMt; ++mt) {
                    const int r0 = warp_m * Traits::kWarpM + group + mt * 16;
                    const float* tile_acc = acc[nt][mt];
                    // Two bf16x2 stores per accumulator tile: rows g and
                    // g+8 of the m16n8 output, columns tig*2/tig*2+1 inside
                    // one 16B chunk.
                    const int off = col & 7;  // element offset in the chunk
                    *reinterpret_cast<__nv_bfloat162*>(
                        out_chunk(r0, col >> 3) + off) =
                        __floats2bfloat162_rn(tile_acc[0] * output_scale + b0,
                                              tile_acc[1] * output_scale + b1);
                    *reinterpret_cast<__nv_bfloat162*>(
                        out_chunk(r0 + 8, col >> 3) + off) =
                        __floats2bfloat162_rn(tile_acc[2] * output_scale + b0,
                                              tile_acc[3] * output_scale + b1);
                }
            }
        } else {
            // Transposed scatter: accumulator (kernel row r0, col) is
            // D[col0_global + col][row0_global + r0], staged at T[col][r0].
            // The acc pair spans two staged rows, so these are scalar
            // stores (the swap path is the rare NN layout). OOB elements
            // store dead lanes of the tile, never copied out.
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = local_col0 + nt * 8;
#pragma unroll
                for (int mt = 0; mt < kMt; ++mt) {
                    const int r0 = warp_m * Traits::kWarpM + group + mt * 16;
                    const int64_t grow = bias_row0 + r0;
                    const float b =
                        bias && grow < m ? __bfloat162float(bias[grow]) : 0.0f;
                    const float* tile_acc = acc[nt][mt];
                    *out_elem(col, r0) =
                        __float2bfloat16(tile_acc[0] * output_scale + b);
                    *out_elem(col + 1, r0) =
                        __float2bfloat16(tile_acc[1] * output_scale + b);
                    *out_elem(col, r0 + 8) =
                        __float2bfloat16(tile_acc[2] * output_scale + b);
                    *out_elem(col + 1, r0 + 8) =
                        __float2bfloat16(tile_acc[3] * output_scale + b);
                }
            }
        }
    }

    // Coalesced copy-out: thread -> one 16B chunk; consecutive threads walk
    // a row so each global transaction covers a full 128B line. Under the
    // swap the staged rows are D-rows counted from block_n's stripe while
    // the row length is kernel m', so row/stride flip to the swapped dims.
    __device__ __forceinline__ void store(__nv_bfloat16* out_bf16) const {
        constexpr int kTotalChunks =
            kBlockM * (kBlockN / 8);  // == kBlockN * (kBlockM/8)
        const int64_t row0_global = block_m * kBlockM;
        const int64_t col0_global = block_n * kBlockN;
        for (int idx = threadIdx.x; idx < kTotalChunks; idx += kCtaThreads) {
            const int r = idx / row_chunks;
            const int c = idx % row_chunks;
            const uint4 v = *reinterpret_cast<const uint4*>(out_chunk(r, c));
            const int64_t row = t_out ? (int64_t)block_n * kBlockN + r
                                      : row0_global + r;
            const int64_t col = t_out ? row0_global + (int64_t)c * 8
                                      : col0_global + (int64_t)c * 8;
            const int64_t rows_total = t_out ? n : m;
            const int64_t row_stride = t_out ? m : n;
            if (row >= rows_total) break;  // rows are consecutive: nothing left
            auto* dst = out_bf16 + row * row_stride + col;
            if (col + 8 <= row_stride &&
                (reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
                if constexpr (kStreamOut) {
                    // Evict-first streaming store knob: neutral on L20
                    // squares, -3..4% on rects; kept for other SKUs.
                    __stcs(reinterpret_cast<uint4*>(dst), v);
                } else {
                    *reinterpret_cast<uint4*>(dst) = v;
                }
            } else {
                // Row-edge chunk or an odd-stride row base: spill the
                // elements that survive the row edge.
                const __nv_bfloat16* elems =
                    reinterpret_cast<const __nv_bfloat16*>(&v);
                for (int e = 0; e < 8 && col + e < row_stride; ++e)
                    dst[e] = elems[e];
            }
        }
    }

    __device__ __forceinline__ void run(float acc[kNt][kMt][4],
                                        __nv_bfloat16* out_bf16) {
        stage(acc);
        __syncthreads();
        store(out_bf16);
    }

  private:
    static constexpr int kCtaThreads = Traits::kCtaThreads;
};

template <typename Policy>
__global__ void __launch_bounds__(Policy::kCtaThreads, Policy::kMinCtas)
    fp8_gemm_kernel(FP8Params p) {
    using Traits = typename Policy::Traits;
    using Mainloop = Fp8CollectiveMainloop<Policy>;
    using Epilogue = Fp8CollectiveEpilogue<Policy>;
    // Stages live in dynamic shared memory so deep pipelines (> 48KB
    // static limit) opt in via cudaFuncSetAttribute in the launcher.
    extern __shared__ __align__(16) char fp8_gemm_smem[];

    // Batch slice (grid.z): broadcast operands carry a 0 stride, so the
    // same pointer serves every batch.
    using T8 = typename Mainloop::T8;
    const T8* a = reinterpret_cast<const T8*>(p.a_ptr) +
                  (int64_t)blockIdx.z * p.a_batch_stride;
    const T8* b = reinterpret_cast<const T8*>(p.b_ptr) +
                  (int64_t)blockIdx.z * p.b_batch_stride;
    auto* out_bf16 = reinterpret_cast<__nv_bfloat16*>(p.out_ptr) +
                     (int64_t)blockIdx.z * p.out_batch_stride;

    static_assert(Mainloop::kBlockM * Mainloop::kBlockN * 2 <=
                      Mainloop::kARing * Mainloop::kBlockM * Mainloop::kK +
                          Mainloop::kBRing * Mainloop::kBlockN * Mainloop::kK,
                  "output tile must fit the reclaimed operand smem");
    const int2 bn = Fp8GemmTileScheduler<Policy::kGroupRaster>::tile(blockIdx, gridDim);
    Mainloop mainloop(fp8_gemm_smem, a, b, p.m, p.n, p.k, p.a_ld, p.b_ld,
                      threadIdx.x, bn);
    float acc[Mainloop::kNt][Mainloop::kMt][4] = {};  // [nt][mt][acc]
    mainloop.prologue();
    mainloop.accumulate(acc);
    // Drain the pipeline before the epilogue reclaims the operand rings.
    astrai::cp_async_wait_all();
    Epilogue(fp8_gemm_smem, p, bn.x, bn.y, threadIdx.x).run(acc, out_bf16);
}

// ---------------------------------------------------------------------------
// Launchers — pure CUDA (no torch), usable from the binding and pure C tests.
// ---------------------------------------------------------------------------

// SM count of the current device (cached per device; benign init race —
// every writer stores the same value).
inline int device_sm_count() {
    static int cached[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    const bool cacheable = dev >= 0 && dev < 64;
    int sms = cacheable ? cached[dev] : 0;
    if (!sms) {
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        sms = sms > 0 ? sms : 1;
        if (cacheable) cached[dev] = sms;
    }
    return sms;
}

// Launch one kernel instantiation with its shared-memory budget: budgets
// beyond the 48KB static limit opt in once per instantiation via
// cudaFuncSetAttribute. Templated on the kernel *value* (auto NTTP) so
// every instantiation owns its own armed flag — same-signature kernels
// must not share it. A failed opt-in arms nothing, so the launch below
// fails loudly through the caller's error checks.
template <auto Kernel, typename... Args>
void launch_with_smem(int smem_bytes, dim3 grid, dim3 block,
                      cudaStream_t stream, Args... args) {
    if (smem_bytes > 48 * 1024) {
        static bool armed = false;  // per instantiation
        if (!armed) {
            const cudaError_t err = cudaFuncSetAttribute(
                Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes);
            armed = (err == cudaSuccess);
        }
    }
    Kernel<<<grid, block, smem_bytes, stream>>>(args...);
}

// Padding-driven small-CTA rule: m or n <= 64 wastes half a 128-row CTA's
// MMA work, and a non-128-divisible shape drags its edge tiles through the
// predicated generic path — when 64 divides both dims, the 64x64 CTA tiles
// exactly and wins that band.
inline bool small_cta_padding(int64_t m, int64_t n) {
    if (m <= 64 || n <= 64) return true;
    const bool big_div = (m % 128 == 0) && (n % 128 == 0);
    const bool small_div = (m % 64 == 0) && (n % 64 == 0);
    return !big_div && small_div;
}

// Launch configuration — a pure function of the problem (unit-testable
// without a GPU). Raster order is not a plan field: every canonical layout
// runs grouped raster; the plain-raster knob stays available through
// launch_plan's GroupRaster parameter for experiments.
struct Fp8GemmPlan {
    enum class Cta { kSmall64, kNarrow128x64, kBig128 };
    Cta cta;
    bool small_s3;  // kSmall64 only: cp.async pipeline depth (2 vs 3 stages)
};

// crosswise_ops counts the operands taking the direct crosswise load
// (A ColMajor / B RowMajor storage): 0 = dual-congruous NT, 1 = TN and the
// NN swap, 2 = TT. The layout shifts the crossovers: the small CTA hides
// the crosswise LDG+PRMT latency far better, while the big CTA's operand
// reuse buys back load bandwidth the crosswise path does not traffic in.
inline Fp8GemmPlan plan_gemm(const FP8Params& p, int crosswise_ops = 0) {
    const int64_t sm = device_sm_count();
    const int64_t tiles_128 =
        (int64_t)p.batch * ((p.m + 127) / 128) * ((p.n + 127) / 128);
    const auto small = [&](bool s3) {
        return Fp8GemmPlan{Fp8GemmPlan::Cta::kSmall64, s3};
    };
    const auto big = [] {
        return Fp8GemmPlan{Fp8GemmPlan::Cta::kBig128, false};
    };
    const auto narrow = [] {
        return Fp8GemmPlan{Fp8GemmPlan::Cta::kNarrow128x64, false};
    };
    // Padding rules first: predication waste beats any wave-fill effect.
    if (small_cta_padding(p.m, p.n)) return small(crosswise_ops > 0);
    if (crosswise_ops > 0) {
        // Crosswise ladder (L20 measured): the small s3 CTA holds ~3/4 of
        // the big CTA's per-SM throughput but tiles 4x finer, so it owns
        // the whole sub-wave band and past it; the big CTA takes over once
        // its grid fills ~1.5 waves.
        if (tiles_128 >= sm * 3 / 2) return big();
        return small(true);
    }
    if (tiles_128 >= sm) {
        // Wave band: pick by the wave-quantization cost ceil(tiles/sm) *
        // T_tile. The narrow tile carries half the big tile's MMA work at
        // ~94% of its per-SM efficiency (T_narrow ~= 0.53 * T_big,
        // integer-scaled by 100 below) — reproduces every measured
        // crossover.
        const int64_t tiles_narrow =
            (int64_t)p.batch * ((p.m + 127) / 128) * ((p.n + 63) / 64);
        const auto waves = [sm](int64_t tiles) { return (tiles + sm - 1) / sm; };
        if (waves(tiles_narrow) * 53 < waves(tiles_128) * 100) return narrow();
        return big();
    }
    // Sub-wave band: the narrow CTA fills the wave with N-tiles at full
    // warp depth once its grid passes ~3/8 of a wave; below that the plain
    // 64x64 CTA's extra parallelism wins, and past ~5/8 of a wave of
    // 128x128 tiles the big CTA's operand reuse wins instead.
    if (tiles_128 >= sm * 5 / 8) return big();
    const int64_t tiles_narrow =
        (int64_t)p.batch * ((p.m + 127) / 128) * ((p.n + 63) / 64);
    if (tiles_narrow >= sm * 3 / 8) return narrow();
    // Full-ring small CTAs: the 24KB s2 variant keeps 4 CTAs/SM while the
    // whole grid stays resident; past that the 32KB s3 variant's deeper
    // pipeline wins on multi-wave grids.
    const int64_t tiles_64 =
        (int64_t)p.batch * ((p.m + 63) / 64) * ((p.n + 63) / 64);
    return small(tiles_64 > sm * 3);
}

// Grid + launch for one concrete Policy — the only place a GEMM kernel goes
// to the wire.
template <typename Policy>
void launch_policy(const FP8Params& p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    dim3 grid((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
              (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
    launch_with_smem<fp8_gemm_kernel<Policy>>(
        Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p);
}

// Plan -> Policy: the production-tuned configs. Big CTA: 128x128 of 8 warps
// x 64x32, kK=64, 2-stage full ring, fast loop only for dual-congruous
// layouts. Narrow: 128x64. Small CTA: 64x64 of 4 warps x 32x32, kK=64,
// kFastLoop always on.
template <FP8Format Fmt, typename LayoutA, typename LayoutB, int GroupRaster>
void launch_plan(const FP8Params& p, const Fp8GemmPlan& plan,
                 cudaStream_t stream) {
    constexpr bool kBigFast = !std::is_same_v<LayoutA, ColMajor> &&
                              !std::is_same_v<LayoutB, RowMajor>;
    switch (plan.cta) {
    case Fp8GemmPlan::Cta::kBig128: {
        using Policy =
            Fp8GemmPolicy<Fmt, 128, 128, LayoutA, LayoutB, 64, 32, 64, 2,
                          GroupRaster, false, kBigFast>;
        launch_policy<Policy>(p, stream);
        break;
    }
    case Fp8GemmPlan::Cta::kNarrow128x64: {
        using Policy =
            Fp8GemmPolicy<Fmt, 128, 64, LayoutA, LayoutB, 32, 32, 64, 2,
                          GroupRaster, false, true>;
        launch_policy<Policy>(p, stream);
        break;
    }
    case Fp8GemmPlan::Cta::kSmall64: {
        if (plan.small_s3) {
            using Policy = Fp8GemmPolicy<Fmt, 64, 64, LayoutA, LayoutB, 32, 32,
                                         64, 3, GroupRaster, false, true>;
            launch_policy<Policy>(p, stream);
        } else {
            using Policy = Fp8GemmPolicy<Fmt, 64, 64, LayoutA, LayoutB, 32, 32,
                                         64, 2, GroupRaster, false, true>;
            launch_policy<Policy>(p, stream);
        }
        break;
    }
    }
}

// Pure problem rewrite: the dual-N-contiguous problem (trans_a/trans_b both
// false) has no dedicated instantiation — it runs as its transpose
// E[N][M] = B^T @ A^T (CUTLASS-sm90's is_swapAB) over swapped operands,
// with p.out_transposed making the epilogue scatter into the caller's
// [M][N] row-major buffer. The rewritten trans flags become the layout tags
// the launcher instantiates; the NN path pays a scalar-store scatter, which
// its rare usage makes the right trade.
inline void canonicalize_gemm(FP8Params& p, bool& trans_a, bool& trans_b) {
    if (!trans_a && !trans_b) {
        FP8Params s = p;  // E = B^T * A^T: swap roles, M <-> N
        s.m = p.n;
        s.n = p.m;
        s.a_ptr = p.b_ptr;
        s.b_ptr = p.a_ptr;
        s.a_ld = p.b_ld;
        s.b_ld = p.a_ld;
        s.a_batch_stride = p.b_batch_stride;
        s.b_batch_stride = p.a_batch_stride;
        s.out_transposed = 1;
        p = s;
        trans_a = trans_b = true;
    }
}

// Entry point: canonicalize the problem, plan the launch, wire the layout
// tags through.
template <FP8Format Fmt>
void gemm(FP8Params p, cudaStream_t stream, bool trans_a, bool trans_b) {
    canonicalize_gemm(p, trans_a, trans_b);
    // Crosswise operand count for the plan: transposed-A storage (ColMajor)
    // and plain-B storage (RowMajor) both take the direct crosswise load.
    const int crosswise = (trans_a ? 1 : 0) + (trans_b ? 0 : 1);
    const Fp8GemmPlan plan = plan_gemm(p, crosswise);
    if (trans_a && trans_b)
        launch_plan<Fmt, ColMajor, ColMajor, 8>(p, plan, stream);
    else if (trans_b)
        launch_plan<Fmt, RowMajor, ColMajor, 8>(p, plan, stream);
    else
        launch_plan<Fmt, ColMajor, RowMajor, 8>(p, plan, stream);
}

}  // namespace fp8
}  // namespace astrai
