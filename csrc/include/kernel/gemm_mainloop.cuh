#pragma once
/*
 * Collective mainloop: smem stage rings, stage loads (congruous cp.async /
 * crosswise LDG+PRMT), per-lane ldmatrix addressing, software-pipelined
 * mma.sync. Addressing scheme + fast-loop peel rationale:
 * docs/developer/kernels/gemm.md.
 */

#include <type_traits>

#include <datatype/dequant.cuh>
#include <memory/load_async.cuh>
#include <memory/load_crosswise.cuh>
#include <memory/load_crosswise_packed.cuh>
#include <memory/pipeline.cuh>
#include <memory/tma.cuh>
#include <mma/mma.cuh>
#include <policy.cuh>
#include <utils/define.cuh>
#include <api/gemm_common.h>
#include <utils/tensor.cuh>

namespace astrai {
namespace gemm {

/*
 * TMA producer context: operand descriptors, ring-slot mbarriers, batch
 * coordinate bits. 2*depth bars — full[0..D): count 1, tripped by the elected
 * thread's expect_tx + the TMA's transaction bytes; empty[D..2D): count =
 * CTA threads, tripped when every consumer finished the slot. The CUTLASS
 * PipelineTmaAsync handshake replaces the per-k-tile __syncthreads: warps
 * skew freely, the producer's overwrite gate is the empty barrier alone.
 * Operand rank is a template bit (strided = 3, broadcast = 2) so the 2D/3D
 * issue pick compiles away.
 */
template <bool kRank3A = false, bool kRank3B = false> struct GemmTmaContext {
    const void* map_a = nullptr;
    const void* map_b = nullptr;
    uint64_t* bars = nullptr;
    int depth = 0; // ring slots (kStages + 1): 2*depth barriers
    int z = 0;     // batch coordinate (rank-3 descriptors)

    DEVICE_FORCEINLINE uint64_t* full(int slot) const { return bars + slot; }
    DEVICE_FORCEINLINE uint64_t* empty(int slot) const { return bars + depth + slot; }
};

template <typename Policy> struct GemmCollectiveMainloop {
    using Traits = typename Policy::Traits;
    using LayoutA = typename Policy::LayoutTagA;
    using LayoutB = typename Policy::LayoutTagB;
    using Smem = GemmSmem<Traits, LayoutA, LayoutB>;
    static constexpr bool kUseTma = Policy::kUseTma;
    static_assert(!kUseTma || (!Smem::kDirectA && !Smem::kDirectB),
                  "TMA staging is congruous-only");
    /*
     * The mma runs on the promoted MmaT (policy/traits.cuh); a lone int8/fp8 side
     * expands in-register between smem read and mma (kDequant* marks the
     * insert, dequant.cuh). AccT rides the mma cell: fp32, int32 for s8.
     */
    using ElemA = typename Traits::ElemA;
    using ElemB = typename Traits::ElemB;
    using MmaT = typename Traits::MmaT;
    using MmaOp = typename Traits::MmaOp;
    using AccT = typename Traits::AccT;
    static constexpr bool kDequantA = Traits::kDequantA;
    static constexpr bool kDequantB = Traits::kDequantB;
    using DequantA = quant::DequantPair<ElemA, MmaT>;
    using DequantB = quant::DequantPair<ElemB, MmaT>;
    static constexpr int kBlockM = Traits::kBlockM;
    static constexpr int kBlockN = Traits::kBlockN;
    static constexpr int kK = Traits::kK;
    static constexpr int kStages = Traits::kStages;
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr bool kDirectA = Smem::kDirectA;
    static constexpr bool kDirectB = Smem::kDirectB;
    /*
     * Crosswise staging splits by width: 16-bit cp.async into the transposed
     * [kK][rows] tile (ldmatrix.trans, pipelineable); 8-bit keeps the
     * synchronous LDG+PRMT path into the canonical tile (no trans ldmatrix).
     */
    static constexpr bool kSyncA = kDirectA && sizeof(ElemA) == 1;
    static constexpr bool kTransA = kDirectA && sizeof(ElemA) == 2;
    static constexpr bool kSyncB = kDirectB && sizeof(ElemB) == 1;
    static constexpr bool kTransB = kDirectB && sizeof(ElemB) == 2;
    /*
     * Dequant inserts are int8-storage only; a dequantized side never rides
     * the 16-bit-only trans staging — kTrans* is already false there.
     */
    static_assert(!kDequantA || sizeof(ElemA) == 1, "in-register dequant targets 1-byte storage");
    static_assert(!kDequantB || sizeof(ElemB) == 1, "in-register dequant targets 1-byte storage");
    static_assert(kStages >= 1 && kStages <= 8, "FP8 GEMM stages must be in [1, 8]");
    /*
     * CTA = (BlockM/WarpM) x (BlockN/WarpN) warps, each warp computing
     * kMt x kNt m16n8k{kMmaK} MMAs. Rings rotate kStages+1 buffers (see
     * GemmSmem) — one __syncthreads per k-tile.
     */
    static constexpr int kMt = Traits::kMt;          // 16-row MMA tiles per warp
    static constexpr int kNt = Traits::kNt;          // 8-col MMA tiles per warp
    static constexpr int kSegs = kK / Traits::kMmaK; // mma-sized k segments
    static constexpr int kARing = Smem::kRingDepth;
    static constexpr int kBRing = Smem::kRingDepth;

    /*
     * Staging layouts, cute-style: composition(Swizzle, Layout<Shape,
     * Stride>) over the row-major 16B-chunk grid (utils/swizzle.cuh), one
     * instance shared by loaders, fragment reads and the lane-offset mirrors
     * below. Canonical [rows][kK] serves congruous + 8-bit crosswise (TMA
     * SWIZZLE_128B / SWIZZLE_64B); trans [kK][rows] the 16-bit crosswise
     * cp.async + ldmatrix.trans (XOR by k-row bits, capped at 8 — the LDSM
     * row budget).
     */
    static constexpr int kChunksA = kK / (16 / (int)sizeof(ElemA));
    static constexpr int kChunksB = kK / (16 / (int)sizeof(ElemB));
    static constexpr int kChunksAT = kBlockM / (16 / (int)sizeof(ElemA));
    static constexpr int kChunksBT = kBlockN / (16 / (int)sizeof(ElemB));
    using SmemLayoutA =
        decltype(composition(Swizzle<log2_const<kChunksA>::value, 3>{},
                             Layout<Shape<kBlockM, kChunksA>, Stride<kChunksA, 1>>{}));
    using SmemLayoutB =
        decltype(composition(Swizzle<log2_const<kChunksB>::value, 3>{},
                             Layout<Shape<kBlockN, kChunksB>, Stride<kChunksB, 1>>{}));
    using SmemLayoutATrans = decltype(composition(
        Swizzle < log2_const<kChunksAT<8 ? kChunksAT : 8>::value, log2_const<kChunksAT>::value>{},
        Layout<Shape<kK, kChunksAT>, Stride<kChunksAT, 1>>{}));
    using SmemLayoutBTrans = decltype(composition(
        Swizzle < log2_const<kChunksBT<8 ? kChunksBT : 8>::value, log2_const<kChunksBT>::value>{},
        Layout<Shape<kK, kChunksBT>, Stride<kChunksBT, 1>>{}));

    /*
     * The k-pair packed grid (8-bit crosswise): staged as 16-bit (row,
     * k-pair) units — packed row j carries the contract pair (2j, 2j+1) — so
     * the 16-bit reader's ldmatrix.trans contract applies unchanged (16
     * packed rows = one mma k-segment). Staged bytes = the canonical tile's,
     * so the ring carve and reclaim budget are untouched.
     */
    static constexpr int kPackChunksA = kBlockM / 8; // 16B chunks per row
    static constexpr int kPackChunksB = kBlockN / 8;
    using SmemLayoutAPack =
        decltype(composition(Swizzle < log2_const<kPackChunksA<8 ? kPackChunksA : 8>::value,
                                                  log2_const<kPackChunksA>::value>{},
                             Layout<Shape<kK / 2, kPackChunksA>, Stride<kPackChunksA, 1>>{}));
    using SmemLayoutBPack =
        decltype(composition(Swizzle < log2_const<kPackChunksB<8 ? kPackChunksB : 8>::value,
                                                  log2_const<kPackChunksB>::value>{},
                             Layout<Shape<kK / 2, kPackChunksB>, Stride<kPackChunksB, 1>>{}));

    /*
     * Packed staging needs: a 1-byte crosswise operand the dequant readers do
     * not own (they read the canonical tile), even kK, power-of-two chunks
     * >= 8 (the swizzle row budget: 8 rows per ldmatrix matrix).
     */
    static constexpr bool kPackOkA = sizeof(ElemA) == 1 && (kK % 2 == 0) && kPackChunksA >= 8 &&
                                     (kPackChunksA & (kPackChunksA - 1)) == 0;
    static constexpr bool kPackOkB = sizeof(ElemB) == 1 && (kK % 2 == 0) && kPackChunksB >= 8 &&
                                     (kPackChunksB & (kPackChunksB - 1)) == 0;
    static constexpr bool kPackA = kDirectA && !kDequantA && kPackOkA;
    static constexpr bool kPackB = kDirectB && !kDequantB && kPackOkB;

    /*
     * One ring type per operand (utils/tensor.cuh): the staged-layout
     * instance each path addresses — trans when 16-bit crosswise staging is
     * active, canonical otherwise (congruous, 8-bit crosswise and dequant
     * readers all use it). Both stagings hold the same element count, so one
     * ring stride serves either; the rings carry slot rotation, byte budgets
     * and the typed tile view — consumers read them off the type.
     */
    using StagedLayoutA =
        std::conditional_t<kPackA,
                           SmemLayoutAPack,
                           std::conditional_t<kTransA, SmemLayoutATrans, SmemLayoutA>>;
    using StagedLayoutB =
        std::conditional_t<kPackB,
                           SmemLayoutBPack,
                           std::conditional_t<kTransB, SmemLayoutBTrans, SmemLayoutB>>;
    using RingA = Tensor<PtrEngine<ElemA>, RingLayout<StagedLayoutA, kARing>>;
    using RingB = Tensor<PtrEngine<ElemB>, RingLayout<StagedLayoutB, kBRing>>;
    using TileA = Tensor<PtrEngine<ElemA>, StagedLayoutA>;
    using TileB = Tensor<PtrEngine<ElemB>, StagedLayoutB>;

    /*
     * The epilogue scatters the output tile into the reclaimed operand rings
     * — the tile must fit them; both orchestrators static_assert this (the
     * launchers' reclaim_fits prices the same rule).
     */
    static constexpr bool kOutputReclaimsRings =
        kBlockM * kBlockN * sizeof(typename Policy::OutT) <=
        RingA::Layout::kTotalBytes + RingB::Layout::kTotalBytes;

    /*
     * The warp's accumulator: typed C cells on a (mt, nt) grid — semantic
     * coordinates all the way to the mma (no pointer decay at the fma seam).
     */
    using AccTensor = typename Traits::AccTensor;

    // Per-stage byte strides.
    static constexpr int kAStageBytes = RingA::Layout::kStageBytes;
    static constexpr int kBStageBytes = RingB::Layout::kStageBytes;

    const RingA ring_a; // A's stage ring; B carves right past its end
    const RingB ring_b;
    const ElemA* const a;
    const ElemB* const b;
    const int64_t m, n, k, a_ld, b_ld;
    const int tid;
    const int64_t block_m, block_n;
    const int warp_m, warp_n;
    const int a_row0; // + mt * 16 in the loop
    const int b_row0; // + nt * 8
    const int64_t tile_count;
    /*
     * Interior-copy verdict, uniform per CTA: whole-CTA, 16B-aligned, K
     * without tail — the mainloop runs the predication-free specialized copy
     * (load_async.cuh's kInterior arm). Measured interleaved 2026-09-16: the
     * specialized copy wins everywhere it applies (an earlier sm_89-era
     * 128x128 regression claim is obsolete — that tile axis is removed).
     */
    const bool use_interior_copy;

    __device__ GemmCollectiveMainloop(char* smem,
                                      const ElemA* a,
                                      const ElemB* b,
                                      int64_t m,
                                      int64_t n,
                                      int64_t k,
                                      int64_t a_ld,
                                      int64_t b_ld,
                                      int tid,
                                      int2 block)
        : ring_a(astrai::make_ring<ElemA, StagedLayoutA, kARing>(smem)),
          ring_b(
              astrai::make_ring<ElemB, StagedLayoutB, kBRing>(smem + RingA::Layout::kTotalBytes)),
          a(a), b(b), m(m), n(n), k(k), a_ld(a_ld), b_ld(b_ld), tid(tid), block_m(block.x),
          block_n(block.y), warp_m((tid >> 5) / Traits::kWarpsN),
          warp_n((tid >> 5) % Traits::kWarpsN), a_row0(warp_m * Traits::kWarpM),
          b_row0(warp_n * Traits::kWarpN), tile_count((k + kK - 1) / kK),
          use_interior_copy(!kSyncA && !kSyncB && ((int64_t)block.x * kBlockM + kBlockM <= m) &&
                            ((int64_t)block.y * kBlockN + kBlockN <= n) &&
                            ((reinterpret_cast<uintptr_t>(a) | (uint64_t)a_ld) & 15) == 0 &&
                            ((reinterpret_cast<uintptr_t>(b) | (uint64_t)b_ld) & 15) == 0 &&
                            (k % kK) == 0) {}

    /*
     * Stage-load one k-tile; each operand picks its loader by staging class
     * (tiles arrive typed by the ring's staged layout, so a mismatched
     * loader/tile pairing is a compile error). kInterior = predication-free
     * interior copy (async phase only — trans qualifies, it is cp.async).
     * kSyncPhase = the synchronous direct loads (8-bit crosswise only), run
     * right after barrier 1 so LDG latency + PRMT transpose overlap the MMA
     * phase instead of stalling the inter-barrier window; false = async
     * loads (kInterior applies; the generic loop runs them after the MMA
     * phase).
     */
    template <bool kInterior = false, bool kSyncPhase = false>
    DEVICE_FORCEINLINE void load_stage(TileA a_tile, TileB b_tile, int64_t k_base) const {
        if constexpr (kSyncPhase) {
            /*
             * 8-bit crosswise: the k-pair packed grid when the tile admits it,
             * the canonical-tile register carry otherwise.
             */
            if constexpr (kPackA)
                load_crosswise_paired<StagedLayoutA, ElemA, kCtaThreads>(a_tile, a, m, k, a_ld, tid,
                                                                         k_base, block_m * kBlockM);
            else if constexpr (kSyncA)
                load_crosswise_direct<StagedLayoutA, ElemA, kCtaThreads>(a_tile, a, m, k, a_ld, tid,
                                                                         k_base, block_m * kBlockM);
            if constexpr (kPackB)
                load_crosswise_paired<StagedLayoutB, ElemB, kCtaThreads>(b_tile, b, n, k, b_ld, tid,
                                                                         k_base, block_n * kBlockN);
            else if constexpr (kSyncB)
                load_crosswise_direct<StagedLayoutB, ElemB, kCtaThreads>(b_tile, b, n, k, b_ld, tid,
                                                                         k_base, block_n * kBlockN);
        } else {
            /*
             * Everything but the 8-bit crosswise (kSync) staging rides
             * cp.async: congruous goes canonical, 16-bit crosswise goes
             * transposed.
             */
            if constexpr (!kSyncA)
                load_operand_tile<StagedLayoutA, ElemA, kCtaThreads, kTransA, kInterior>(
                    a_tile, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
            if constexpr (!kSyncB)
                load_operand_tile<StagedLayoutB, ElemB, kCtaThreads, kTransB, kInterior>(
                    b_tile, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
        }
    }

    /*
     * One TMA stage issue, elected-thread only: gate on the slot's empty
     * barrier (skipped on the ring's first sweep), arm the full barrier's
     * byte count, issue both operand boxes (rank rides the context type).
     */
    template <bool kRank3A, bool kRank3B>
    DEVICE_FORCEINLINE void tma_issue_stage(const GemmTmaContext<kRank3A, kRank3B>& tma,
                                            int tile) const {
        const int slot = tile % kARing;
        if (tile >= kARing)
            astrai::mbarrier_wait_parity(tma.empty(slot), (uint32_t)(((tile / kARing) - 1) & 1));
        uint64_t* bar = tma.full(slot);
        mbarrier_arrive_expect_tx(bar, (uint32_t)((uint64_t)kAStageBytes + kBStageBytes));
        const int x = (int)((int64_t)tile * kK);
        astrai::tma_load<kRank3A>(tma.map_a, bar, astrai::stage_of(ring_a, tile).engine.ptr,
                                  x * (int)sizeof(ElemA), (int)(block_m * kBlockM), tma.z);
        astrai::tma_load<kRank3B>(tma.map_b, bar, astrai::stage_of(ring_b, tile).engine.ptr,
                                  x * (int)sizeof(ElemB), (int)(block_n * kBlockN), tma.z);
    }

    /*
     * Prime the pipeline: kStages committed groups, one per slot. Commits
     * are unconditional — short-K skipped stages commit empty groups, so the
     * group sequence stays tile-indexed and the steady-state wait needs no
     * runtime dispatch. TMA arms only stages that carry a copy: an expect_tx
     * barrier with no transaction never trips, so short-K slots are never
     * waited on.
     */
    template <bool kRank3A = false, bool kRank3B = false>
    DEVICE_FORCEINLINE void prologue(const GemmTmaContext<kRank3A, kRank3B>& tma = {}) const {
        if constexpr (kUseTma) {
            if (tid == 0) {
#pragma unroll
                for (int stage = 0; stage < kStages; ++stage) {
                    if (stage < tile_count)
                        tma_issue_stage(tma, stage);
                }
            }
            return;
        }
        const astrai::PipelineSync<kStages> pipe;
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
            if (stage < tile_count) {
                if (use_interior_copy)
                    load_stage<true>(astrai::stage_of(ring_a, stage),
                                     astrai::stage_of(ring_b, stage), (int64_t)stage * kK);
                else
                    load_stage(astrai::stage_of(ring_a, stage), astrai::stage_of(ring_b, stage),
                               (int64_t)stage * kK);
                load_stage<false, true>(astrai::stage_of(ring_a, stage),
                                        astrai::stage_of(ring_b, stage), (int64_t)stage * kK);
            }
            pipe.producer_commit();
        }
    }

    /*
     * Steady-state mainloop, specialized on kInterior: interior copy runs
     * predication-free with loop-carried pointers, generic keeps full
     * predication. kTma swaps the discipline: per-thread cp.async chunks +
     * wait_group/syncthreads fence become one elected-thread TMA issue and an
     * mbarrier phase wait (the CTA barrier stays the slot-release guarantee:
     * every thread finished reading tile i-1 before i+kStages overwrites).
     */
    template <bool kInterior, bool kTma = false, bool kRank3A = false, bool kRank3B = false>
    DEVICE_FORCEINLINE void run_loop(AccTensor& acc,
                                     const GemmTmaContext<kRank3A, kRank3B>& tma = {}) const {
        const astrai::PipelineSync<kStages> pipe;
        const int lane = tid & 31;
        /*
         * Fast-path write carries: one per congruous operand (crosswise gets
         * the no-op type), targeting the first prefetched tile (kStages).
         * Read carries: the LDSM base with the lane offset folded in,
         * advanced one stage per iteration with an equality wrap — replaces
         * the per-tile (tile % ring) * stage_bytes recomputation (a
         * UIMAD.WIDE magic-division ladder in SASS). Carries ride the rings,
         * so each carries its staging geometry.
         */
        PrefetchCarry<!kSyncA && !kTma, RingA, kCtaThreads, kTransA> carry_a(
            ring_a, a, a_ld, block_m * kBlockM, tid, kStages);
        PrefetchCarry<!kSyncB && !kTma, RingB, kCtaThreads, kTransB> carry_b(
            ring_b, b, b_ld, block_n * kBlockN, tid, kStages);
        /*
         * The 8-bit crosswise operands' register carry: issue() runs the global
         * runs before the MMA phase, commit() the byte-perm and STS after it.
         * Packed-grid operands ride the same seam with the two-run sibling.
         */
        std::conditional_t<kPackA, PairPackCarry<StagedLayoutA, ElemA, kCtaThreads, true>,
                           CrosswiseCarry<StagedLayoutA, ElemA, kCtaThreads, kSyncA>>
            sync_a;
        std::conditional_t<kPackB, PairPackCarry<StagedLayoutB, ElemB, kCtaThreads, true>,
                           CrosswiseCarry<StagedLayoutB, ElemB, kCtaThreads, kSyncB>>
            sync_b;
        const unsigned a_rd0 = __cvta_generic_to_shared(ring_a.engine.ptr) +
                               (kPackA ? a_pack_lane_off(lane)
                                       : (kTransA ? a_trans_lane_off(lane) : a_lane_off(lane)));
        const unsigned b_rd0 =
            __cvta_generic_to_shared(ring_b.engine.ptr) +
            (kPackB ? b_pack_lane_off(lane)
                    : (kTransB ? b_trans_lane_off(lane)
                               : (kPairB ? b4_lane_off(lane) : b_lane_off(lane))));
        const unsigned a_rd_end = a_rd0 + (unsigned)(kARing * kAStageBytes);
        const unsigned b_rd_end = b_rd0 + (unsigned)(kBRing * kBStageBytes);
        unsigned a_rd = a_rd0, b_rd = b_rd0;
        for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            /*
             * Steady state: exactly kStages-1 younger groups in flight when
             * this fires; the tail's unconditional empty commits keep it true.
             */
            const bool prefetch = tile_index + kStages < tile_count;
            /*
             * Steady-state wait: TMA waits the slot's full barrier (phase
             * flips once per sweep) — no CTA-wide barrier; each warp releases
             * its slot below after its last fragment read, the producer's
             * overwrite gate is the empty barrier alone. cp.async drains its
             * group ladder then joins the CTA (the join doubles as release).
             */
            if constexpr (kTma) {
                astrai::mbarrier_wait_parity(tma.full((int)(tile_index % kARing)),
                                             static_cast<uint32_t>((tile_index / kARing) & 1));
            } else {
                pipe.consumer_wait();
            }

            /*
             * Staging for tile i+kStages (its slot = (i-1)'s, released above):
             * elected thread arms + issues both TMA boxes; 8-bit crosswise
             * issues its LDG.128 runs here (transpose + STS follow the MMA
             * phase, so global latency overlaps tensor-pipe work).
             */
            if constexpr (kTma) {
                if (prefetch && tid == 0)
                    tma_issue_stage(tma, (int)(tile_index + kStages));
            } else if (prefetch) {
                if constexpr (kSyncA)
                    sync_a.issue(a, m, k, a_ld, tid, (tile_index + kStages) * kK,
                                 block_m * kBlockM);
                if constexpr (kSyncB)
                    sync_b.issue(b, n, k, b_ld, tid, (tile_index + kStages) * kK,
                                 block_n * kBlockN);
            }

            const unsigned a_addr = a_rd;
            const unsigned b_addr = b_rd;
            /*
             * Per-k_seg base pair (cuBLAS's scheme): seg s = seg-0 base XOR
             * (s * kSegXor) — one LOP3 per extra seg, never per fragment.
             * Trans tiles step a plain ADD (k rows at fixed stride). Every
             * LDSM below addresses [base + immediate].
             */
            unsigned a_seg[kSegs], b_seg[kSegs];
#pragma unroll
            for (int s = 0; s < kSegs; ++s) {
                a_seg[s] = (kTransA || kPackA) ? (a_addr + (unsigned)(s * kTransSegA))
                                               : (a_addr ^ (unsigned)(s * kSegXorA));
                b_seg[s] = (kTransB || kPackB) ? (b_addr + (unsigned)(s * kTransSegB))
                                               : (b_addr ^ (unsigned)(s * kSegXorB));
            }

            /*
             * kNt ldmatrix.x2 (B) + kMt ldmatrix.x4 (A) feed kMt*kNt*2
             * mma.sync per k_seg — 0.5 loads per MMA. B fragments double-
             * buffer across k_segs; kPairB folds two adjacent nt fragments
             * into one x4 (b4_lane_off). Fragment arrays hold typed cells:
             * loads fill, mma consumes by reference.
             */
            typename MmaOp::BFrag b_frag[2][kNt];
            BFragPair b_frag4[2][kNt / 2];
            /*
             * Staged tensor hoisted out of the k_seg/mt loops: the slot pick
             * is a modulo and regressed the dequant readers' register budget.
             */
            const auto b_tile = astrai::stage_of(ring_b, tile_index);
            load_b_frags_at(b_frag[0], b_frag4[0], b_tile, 0, b_seg[0], lane);
#pragma unroll
            for (int k_seg = 0; k_seg < kSegs; ++k_seg) {
                const int bcur = k_seg & 1, bnext = bcur ^ 1;
                if (k_seg + 1 < kSegs)
                    load_b_frags_at(b_frag[bnext], b_frag4[bnext], b_tile, k_seg + 1,
                                    b_seg[k_seg + 1], lane);
                /*
                 * Software-pipelined A fragments: row mt+1's ldmatrix.x4 issues
                 * before row mt's MMAs so LDS latency hides behind tensor-pipe
                 * work (costs 4 registers). Trans tiles advance the m window by
                 * XOR, canonical by the 16-row stride. Dequant A (W8A8) fills
                 * ALL m-row fragments upfront — the pipelined ldmatrix would
                 * clobber converted fragments with raw 2-byte-layout data.
                 */
                typename MmaOp::AFrag a_frag[kMt + 1];
                if constexpr (kDequantA) {
                    const auto a_tile = astrai::stage_of(ring_a, tile_index);
#pragma unroll
                    for (int mt = 0; mt < kMt; ++mt)
                        load_a_frags_at(a_frag[mt], a_tile, k_seg, mt, lane);
                } else if constexpr (kTransA || kPackA) {
                    astrai::ldmatrix_x4_lane<true>(a_frag[0], a_seg[k_seg]);
                } else {
                    astrai::ldmatrix_x4_lane(a_frag[0], a_seg[k_seg]);
                }
#pragma unroll
                for (int mt = 0; mt < kMt; ++mt) {
                    if constexpr (!kDequantA) {
                        if (mt + 1 < kMt) {
                            const unsigned a_next =
                                (kTransA || kPackA) ? (a_seg[k_seg] ^ (unsigned)((mt + 1) * kMtXor))
                                                    : (a_seg[k_seg] + (mt + 1) * kMtStep);
                            if constexpr (kTransA || kPackA)
                                astrai::ldmatrix_x4_lane<true>(a_frag[mt + 1], a_next);
                            else
                                astrai::ldmatrix_x4_lane(a_frag[mt + 1], a_next);
                        }
                    }
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt) {
                        if constexpr (kPairB)
                            MmaOp::fma(*acc(mt, nt), a_frag[mt],
                                       b_frag4[bcur][nt >> 1].cell(nt & 1), *acc(mt, nt));
                        else
                            MmaOp::fma(*acc(mt, nt), a_frag[mt], b_frag[bcur][nt], *acc(mt, nt));
                    }
                }
                /*
                 * Next tile's LDGSTS chunks inside the MMA phase: A's after the
                 * first k_seg's MMA batch, B's after the last.
                 */
                if constexpr (kInterior && !kTma) {
                    if (k_seg == 0)
                        carry_a.emit(prefetch);
                    if (k_seg == kSegs - 1)
                        carry_b.emit(prefetch);
                }
            }
            /*
             * Generic loop (no interleaved prefetch): the next tile's
             * predicated loads run after the MMA phase — congruous cp.async
             * chunks, 8-bit crosswise transpose + STS of the issued runs.
             */
            if constexpr (!kInterior && !kTma) {
                if (prefetch) {
                    load_stage(astrai::stage_of(ring_a, tile_index + kStages),
                               astrai::stage_of(ring_b, tile_index + kStages),
                               (tile_index + kStages) * kK);
                    if constexpr (kSyncA)
                        sync_a.commit(astrai::stage_of(ring_a, tile_index + kStages), a, m, k, a_ld,
                                      tid, (tile_index + kStages) * kK, block_m * kBlockM);
                    if constexpr (kSyncB)
                        sync_b.commit(astrai::stage_of(ring_b, tile_index + kStages), b, n, k, b_ld,
                                      tid, (tile_index + kStages) * kK, block_n * kBlockN);
                }
            }
            /*
             * Unconditional commit: empty in the tail, pads the group sequence
             * so the fixed wait stays correct. TMA commits via tma_issue_stage.
             */
            if constexpr (!kTma)
                pipe.producer_commit();
            /*
             * TMA consumer release: this thread's reads are done; the empty
             * barrier trips once every thread arrives, gating the overwrite.
             */
            if constexpr (kTma)
                astrai::mbarrier_arrive(tma.empty((int)(tile_index % kARing)));
            a_rd += (unsigned)kAStageBytes;
            if (a_rd == a_rd_end)
                a_rd = a_rd0;
            b_rd += (unsigned)kBStageBytes;
            if (b_rd == b_rd_end)
                b_rd = b_rd0;
            if constexpr (kInterior && !kTma) {
                carry_a.advance();
                carry_b.advance();
            }
        }
    }

    template <bool kRank3A = false, bool kRank3B = false>
    DEVICE_FORCEINLINE void accumulate(AccTensor& acc,
                                       const GemmTmaContext<kRank3A, kRank3B>& tma = {}) const {
        if constexpr (kUseTma) {
            run_loop<false, true, kRank3A, kRank3B>(acc, tma);
        } else {
            if (use_interior_copy)
                run_loop<true>(acc);
            else
                run_loop<false>(acc);
        }
    }

  private:
    /*
     * One ldmatrix.x4 payload covering two adjacent n8 B fragments (the
     * kPairB fold): cell(i) picks the fragment nt consumes — the pairing
     * lives in the type, not fma-seam pointer arithmetic. (Not "half":
     * nvcc reserves that name for the fp16 type.)
     */
    struct BFragPair : ArrayEngine<unsigned, 4> {
        DEVICE_FORCEINLINE typename MmaOp::BFrag cell(int i) const {
            return {storage[2 * i + 0], storage[2 * i + 1]};
        }
    };

    /*
     * Per-lane ldmatrix fragment addressing (base-pair scheme, mirrored from
     * the cuBLAS SASS; derivation in the design notes): one base register per
     * operand per k_seg, every fragment offset an LDSM immediate — zero
     * address arithmetic inside the MMA phase. Offsets are stage-relative
     * BYTES (the *_lane primitives take raw smem byte addresses; element math
     * scaled by sizeof(ElemT)). The swizzle chunk term comes from the declared
     * staging layouts — the same instances the tiles apply, so the mirror
     * cannot drift. Canonical and trans are one formula per staging,
     * parameterized by layout/ElemT/extent; the wrappers below pick lane bits
     * and base row (A's fragment row carries +8-row and +1-chunk halves, B
     * uses the +8-row bit as its chunk half).
     */
    template <typename SmemLayoutT, typename ElemT>
    static DEVICE_FORCEINLINE unsigned canonical_lane_off(int64_t row, int chunk_half, int lane) {
        constexpr int kChunkShift = log2_const<16 / sizeof(ElemT)>::value;
        const unsigned lswz =
            static_cast<unsigned>(((lane & 7) >> SmemLayoutT::kRowShift) & SmemLayoutT::kMask);
        return static_cast<unsigned>((row * kK + ((chunk_half ^ lswz) << kChunkShift)) *
                                     sizeof(ElemT));
    }

    /*
     * Trans-tile addressing (crosswise 16-bit): the LDSM row is a k line, the
     * 16B chunk a non-contract-dim window, chunks swizzled by k-row bits.
     * ldmatrix.trans lane contract: lanes 0-7 k rows 0-7, 8-15 k rows 8-15
     * (second k half), 16-31 (x4) one column chunk (the +8 half); x2 ignores
     * 16-31. kMtXor/kNtXor: one m/n-tile step in 16B chunks (XOR, not add);
     * kTransSeg*: one mma k-segment = kMmaK k rows.
     */
    template <typename SmemLayoutT, typename ElemT>
    static DEVICE_FORCEINLINE unsigned trans_lane_off(int krow, int col, int block_extent) {
        const unsigned lswz =
            static_cast<unsigned>((krow >> SmemLayoutT::kRowShift) & SmemLayoutT::kMask);
        return static_cast<unsigned>(
            ((int64_t)krow * block_extent + (((col >> 3) ^ lswz) << 3) + (col & 7)) *
            sizeof(ElemT));
    }

    static constexpr int kChunkElems = 16 / sizeof(ElemA);
    static constexpr int kChunkShift = log2_const<kChunkElems>::value;
    DEVICE_FORCEINLINE unsigned a_lane_off(int lane) const {
        /*
         * Stage-relative, loop-invariant per-lane base; A's fragment row
         * carries the +8-row (rh8) and +1-chunk (rh16) halves.
         */
        return canonical_lane_off<SmemLayoutA, ElemA>(a_row0 + ((lane >> 3) & 1) * 8 + (lane & 7),
                                                      lane >> 4, lane);
    }
    DEVICE_FORCEINLINE unsigned b_lane_off(int lane) const {
        // ldmatrix (non-dequant) B addressing: byte offsets in ElemB units.
        return canonical_lane_off<SmemLayoutB, ElemB>(b_row0 + (lane & 7), (lane >> 3) & 1, lane);
    }
    /*
     * x4-paired B loads: one ldmatrix.x4 feeds two adjacent nt fragments.
     * Lane contract: lanes 0-7 rows n0..n7 chunk c, 8-15 rows n0..n7 chunk
     * c+1, 16-23 rows n8..n15 chunk c, 24-31 n8..n15 chunk c+1. The +8-row
     * step never reaches the swizzle source bits for kK <= 64; kK=128
     * swizzles row[2:0] where +8 flips bits, so that config keeps x2 loads.
     * Fragment step constants in BYTES: kSegXor{A,B} = one mma k-segment of
     * the operand's STORAGE type (32B for every ldmatrix-fed dtype), i.e.
     * two 16B chunks; mixed pairs keep a step per side; dequant sides never
     * consume theirs.
     */
    static constexpr unsigned kMtStep = 16u * kK * sizeof(ElemA); // m-tile row step
    static constexpr unsigned kNtStep = 8u * kK * sizeof(ElemB);  // n-tile row step
    static constexpr unsigned kSegXorA = (unsigned)Traits::kMmaK * sizeof(ElemA);
    static constexpr unsigned kSegXorB = (unsigned)Traits::kMmaK * sizeof(ElemB);
    static constexpr bool kPairB = !kDequantB && !kPackB && kK * sizeof(ElemB) / 16 <= 4;
    static_assert(!kPairB || kNt % 2 == 0, "B pairing needs even kNt");
    static_assert(!kPairB || !kTransB, "2-byte crosswise B never pairs (chunk budget)");
    static constexpr unsigned kPairStep = 16u * kK * sizeof(ElemB); // nt-pair row step
    DEVICE_FORCEINLINE unsigned b4_lane_off(int lane) const {
        return b_lane_off(lane) + (lane >> 4) * kPairStep / 2;
    }

    static constexpr unsigned kMtXor = 32u; // m16 = 2 chunks
    static constexpr unsigned kNtXor = 16u; // n8 = 1 chunk
    static constexpr unsigned kTransSegA = (unsigned)Traits::kMmaK * kBlockM * sizeof(ElemA);
    static constexpr unsigned kTransSegB = (unsigned)Traits::kMmaK * kBlockN * sizeof(ElemB);
    DEVICE_FORCEINLINE unsigned a_trans_lane_off(int lane) const {
        /*
         * x4 matrix order must match the mma's A-register order (m+8 rides
         * reg1, k+8 reg2): lanes 8-15 step the m+8 chunk, lanes 16-31 the
         * k+8 row half.
         */
        return trans_lane_off<SmemLayoutATrans, ElemA>((lane & 7) + ((lane >> 4) << 3),
                                                       a_row0 + (((lane >> 3) & 1) << 3), kBlockM);
    }
    DEVICE_FORCEINLINE unsigned b_trans_lane_off(int lane) const {
        // nt windows step by kNtXor at call sites (col stays at b_row0).
        return trans_lane_off<SmemLayoutBTrans, ElemB>((lane & 7) + (((lane >> 3) & 1) << 3),
                                                       b_row0, kBlockN);
    }

    /*
     * Packed-grid lane offsets: the trans formula one width down (a packed
     * row is 16-bit units, 8 per 16B chunk; the row extent is in UNITS). The
     * lane contract, XOR steps and per-k-segment ADD are the trans reader's:
     * 16 packed rows = one mma k-segment.
     */
    DEVICE_FORCEINLINE unsigned a_pack_lane_off(int lane) const {
        return trans_lane_off<SmemLayoutAPack, unsigned short>(
            (lane & 7) + ((lane >> 4) << 3), a_row0 + (((lane >> 3) & 1) << 3), kBlockM);
    }
    DEVICE_FORCEINLINE unsigned b_pack_lane_off(int lane) const {
        return trans_lane_off<SmemLayoutBPack, unsigned short>(
            (lane & 7) + (((lane >> 3) & 1) << 3), b_row0, kBlockN);
    }

    /*
     * Dequantized A fragments (W8A8 activation side): lane l's m16n8k16 A
     * fragment (q = l>>2, c2 = (l&3)*2) holds tile
     * [m = a_row0 + mt*16 + q (+8)][k = k_seg*16 + c2 (+8)] in register order
     * (m, m+8, k+8, m+8&k+8) — matching the ldmatrix x4 order the non-dequant
     * path produces. Both u16 reads of one row stay in one swizzle chunk.
     */
    DEVICE_FORCEINLINE void
    load_a_frags_at(typename MmaOp::AFrag& frag, TileA stage, int k_seg, int mt, int lane) const {
        const int q = lane >> 2, c2 = (lane & 3) * 2;
        const int row = a_row0 + mt * 16 + q;
        const ElemA* p0 = stage(row, k_seg * 16 + c2);
        const ElemA* p8 = stage(row + 8, k_seg * 16 + c2);
        frag[0] = DequantA::pair(*(const unsigned short*)p0);
        frag[1] = DequantA::pair(*(const unsigned short*)p8);
        frag[2] = DequantA::pair(*(const unsigned short*)(p0 + 8));
        frag[3] = DequantA::pair(*(const unsigned short*)(p8 + 8));
    }

    /*
     * One k_seg's B-fragment loads, shared by the initial fill and the
     * double-buffer's next-seg fill; the unused frag2/frag4 is never touched.
     */
    DEVICE_FORCEINLINE void load_b_frags(typename MmaOp::BFrag (&frag2)[kNt],
                                         BFragPair (&frag4)[kNt / 2],
                                         unsigned seg_base) const {
#pragma unroll
        for (int p = 0; p < kNt / 2; ++p) {
            if constexpr (kTransB || kPackB) {
                /*
                 * Trans/packed: each x2.trans reads 16 k rows at one n chunk;
                 * nt windows step one XORed chunk.
                 */
                astrai::ldmatrix_x2_lane<true>(frag2[p * 2], seg_base ^ (unsigned)(p * 2 * kNtXor));
                astrai::ldmatrix_x2_lane<true>(frag2[p * 2 + 1],
                                               seg_base ^ (unsigned)((p * 2 + 1) * kNtXor));
            } else if constexpr (kPairB) {
                astrai::ldmatrix_x4_lane(frag4[p], seg_base + p * kPairStep);
            } else {
                astrai::ldmatrix_x2_lane(frag2[p * 2], seg_base + p * 2 * kNtStep);
                astrai::ldmatrix_x2_lane(frag2[p * 2 + 1], seg_base + (p * 2 + 1) * kNtStep);
            }
        }
    }

    /*
     * Dequantized B fragments (weight side): lane l's m16n8k16 B fragment
     * (quad q = l>>2, r = l&3) holds tile[n = b_row0 + nt*8 + q]
     * [k = k_seg*16 + {2r, 2r+1, 2r+8, 2r+9}] as two packed pairs — both u16
     * reads land in one 16B swizzle chunk, so staged-tile addressing works;
     * the LOP3 expansion (dequant.cuh) is exact for int8.
     */
    DEVICE_FORCEINLINE void load_b_frags_at(typename MmaOp::BFrag (&frag2)[kNt],
                                            BFragPair (&frag4)[kNt / 2],
                                            TileB stage,
                                            int k_seg,
                                            unsigned seg_base,
                                            int lane) const {
        if constexpr (kDequantB) {
            const int q = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int row = b_row0 + nt * 8 + q;
                const ElemB* p0 = stage(row, k_seg * 16 + c2);
                const ElemB* p1 = stage(row, k_seg * 16 + c2 + 8);
                frag2[nt][0] = DequantB::pair(*(const unsigned short*)p0);
                frag2[nt][1] = DequantB::pair(*(const unsigned short*)p1);
            }
        } else {
            load_b_frags(frag2, frag4, seg_base);
        }
    }
};

} // namespace gemm
} // namespace astrai
