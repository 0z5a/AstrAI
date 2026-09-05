#pragma once
// Collective mainloop: shared-memory stage rings, the gmem->smem stage loads
// (congruous cp.async / crosswise LDG+PRMT), the per-lane ldmatrix fragment
// addressing and the software-pipelined mma.sync loop. The fragment
// addressing scheme and the fast-loop peel rationale live in
// docs/developer/cuda_kernels.md.

#include <type_traits>

#include "common/mma.cuh"
#include "gemm/common.h"
#include "load.cuh"
#include "policy.cuh"

namespace astrai {
namespace gemm {

template <typename Policy>
struct GemmCollectiveMainloop {
    using Traits = typename Policy::Traits;
    using LayoutA = typename Policy::LayoutTagA;
    using LayoutB = typename Policy::LayoutTagB;
    using Smem = GemmSmem<Traits, LayoutA, LayoutB>;
    static constexpr bool kFastLoop = Policy::kFastLoop;
    using ElemT = typename Traits::ElemT;  // operand element type via traits
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
    // GemmSmem) — one __syncthreads per k-tile.
    static constexpr int kMt = Traits::kWarpM / 16;  // 16-row MMA tiles per warp
    static constexpr int kNt = Traits::kWarpN / 8;   // 8-col MMA tiles per warp
    static constexpr int kSegs = kK / Traits::kMmaK;  // mma-sized k segments
    static constexpr int kARing = Smem::kRingDepth;
    static constexpr int kBRing = Smem::kRingDepth;
    // Stage strides in BOTH units: ElemT* staging arithmetic uses elements,
    // the smem split / byte-address carries / PrefetchCarry use bytes.
    static constexpr int kAStageElems = kBlockM * kK;
    static constexpr int kBStageElems = kBlockN * kK;
    static constexpr int kAStageBytes = kAStageElems * (int)sizeof(ElemT);
    static constexpr int kBStageBytes = kBStageElems * (int)sizeof(ElemT);

    ElemT* const a_base;
    ElemT* const b_base;
    const ElemT* const a;
    const ElemT* const b;
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

    __device__ GemmCollectiveMainloop(char* smem, const ElemT* a, const ElemT* b,
                                     int64_t m, int64_t n, int64_t k,
                                     int64_t a_ld, int64_t b_ld, int tid,
                                     int2 block)
        : a_base(reinterpret_cast<ElemT*>(smem)),
          b_base(reinterpret_cast<ElemT*>(smem + kARing * kAStageBytes)),
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
    __device__ __forceinline__ ElemT* a_stage_of(int64_t tile) const {
        return a_base + (size_t)(tile % kARing) * kAStageElems;
    }
    __device__ __forceinline__ ElemT* b_stage_of(int64_t tile) const {
        return b_base + (size_t)(tile % kBRing) * kBStageElems;
    }
    // Asynchronous congruous loads for one k-tile: cp.async into the
    // canonical rings; kFast selects the predication-free interior copy
    // (fast_cta admits only congruous operands). Called after the
    // post-compute barrier, alongside the commit.
    template <bool kFast = false>
    __device__ __forceinline__ void load_async(ElemT* a_stage, ElemT* b_stage,
                                               int64_t k_base) const {
        if constexpr (!kDirectA)
            load_operand_tile<ElemT, kK, kBlockM, kCtaThreads, kFast>(
                a_stage, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
        if constexpr (!kDirectB)
            load_operand_tile<ElemT, kK, kBlockN, kCtaThreads, kFast>(
                b_stage, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
    }
    // Synchronous direct-crosswise loads for one k-tile. In the steady
    // state this runs right after barrier 1, so the LDG latency and the
    // PRMT transpose overlap the MMA phase instead of stalling the
    // inter-barrier window.
    __device__ __forceinline__ void load_direct(ElemT* a_stage, ElemT* b_stage,
                                                int64_t k_base) const {
        if constexpr (kDirectA)
            load_crosswise_direct<ElemT, kK, kBlockM, kCtaThreads>(
                a_stage, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
        if constexpr (kDirectB)
            load_crosswise_direct<ElemT, kK, kBlockN, kCtaThreads>(
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
        // prefetched tile (kStages). Steady-state read carries: the LDSM
        // base of the current k-tile's stage with the lane offset folded
        // in, advanced one stage per iteration with an equality wrap —
        // replaces the per-k-tile (tile % ring) * stage_bytes
        // recomputation (a UIMAD.WIDE magic-division ladder in SASS).
        PrefetchCarry<!kDirectA, ElemT, kK, kBlockM, kCtaThreads> carry_a(
            a_base, kARing, kAStageBytes, a, a_ld, block_m * kBlockM, tid,
            kStages);
        PrefetchCarry<!kDirectB, ElemT, kK, kBlockN, kCtaThreads> carry_b(
            b_base, kBRing, kBStageBytes, b, b_ld, block_n * kBlockN, tid,
            kStages);
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
        // base XOR (s * kSegXor) — one LOP3 per extra seg per k-tile, never
        // per fragment. Every LDSM below addresses [base + immediate].
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
                astrai::mma_sync<ElemT>(acc[nt][mt], a_frag[mt], bops,
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
    // from the cuBLAS SASS; derivation in the design notes): one base
    // register per operand per k_seg, every fragment offset an LDSM
    // immediate — zero address arithmetic inside the MMA phase.
    // Per-lane stage-relative BYTE offsets: the ldmatrix *_lane primitives
    // take raw shared-memory byte addresses, so every offset below is
    // element math scaled by sizeof(ElemT). The swizzle chunk term mirrors
    // tile_at (16B chunks = kChunkElems elements).
    static constexpr int kChunkElems = 16 / sizeof(ElemT);
    static constexpr int kChunkShift = log2_const<kChunkElems>::value;
    __device__ __forceinline__ unsigned a_lane_off(int lane) const {
        const int r7 = lane & 7;          // row within the 8-row matrix
        const int rh8 = (lane >> 3) & 1;  // +8 rows (A: lanes 8-15, 24-31)
        const int rh16 = lane >> 4;       // +1 chunk (A: lanes 16-31)
        constexpr int kChunks = kK / kChunkElems;
        constexpr int kShift = 3 - log2_const<kChunks>::value;  // tile_at's shift
        const unsigned lswz =
            static_cast<unsigned>((r7 >> kShift) & (kChunks - 1));
        // Stage-relative, loop-invariant per-lane base; A's fragment row
        // carries the +8-row (rh8) and +1-chunk (rh16) halves.
        return static_cast<unsigned>(((a_row0 + rh8 * 8 + r7) * kK +
                                      ((rh16 ^ lswz) << kChunkShift)) *
                                     sizeof(ElemT));
    }
    __device__ __forceinline__ unsigned b_lane_off(int lane) const {
        const int r7 = lane & 7;
        const int rh8 = (lane >> 3) & 1;  // +8 rows (B uses rh8 as its chunk half)
        constexpr int kChunks = kK / kChunkElems;
        constexpr int kShift = 3 - log2_const<kChunks>::value;
        const unsigned lswz =
            static_cast<unsigned>((r7 >> kShift) & (kChunks - 1));
        return static_cast<unsigned>(((b_row0 + r7) * kK +
                                      ((rh8 ^ lswz) << kChunkShift)) *
                                     sizeof(ElemT));
    }
    // x4-paired B loads: one ldmatrix.x4 feeds the two adjacent nt
    // fragments. Lane contract: lanes 0-7 address rows n0..n7 chunk c,
    // lanes 8-15 rows n0..n7 chunk c+1, lanes 16-23 rows n8..n15 chunk c,
    // lanes 24-31 rows n8..n15 chunk c+1. The +8-row step never reaches
    // the swizzle source bits for kK <= 64; kK=128 swizzles on row[2:0]
    // where +8 flips bits, so that config keeps the x2 loads.
    // Fragment step constants, in BYTES (consumed by the *_lane address
    // math). kSegXor = one mma k-segment = kMmaK elements — 32B for every
    // supported dtype (fp8 k32 x 1B, bf16 k16 x 2B), i.e. two 16B chunks.
    static constexpr unsigned kMtStep = 16u * kK * sizeof(ElemT);  // m-tile row step
    static constexpr unsigned kNtStep = 8u * kK * sizeof(ElemT);   // n-tile row step
    static constexpr unsigned kSegXor = (unsigned)Traits::kMmaK * sizeof(ElemT);
    static constexpr bool kPairB = kK * sizeof(ElemT) / 16 <= 4;
    static_assert(!kPairB || kNt % 2 == 0, "B pairing needs even kNt");
    static constexpr unsigned kPairStep = 16u * kK * sizeof(ElemT);  // nt-pair row step
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

}  // namespace gemm
}  // namespace astrai
