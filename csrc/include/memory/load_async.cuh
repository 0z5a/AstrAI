#pragma once
/* Congruous and 16-bit transposed operand staging through cp.async. */
#include <memory/pipeline.cuh>
#include <utils/define.cuh>
#include <utils/tensor.cuh>

namespace astrai {
namespace gemm {

/*
 * Stage-load one operand tile into its swizzled ring slot; the two
 * geometries are role-swapped mirrors of the SAME loop, so one template
 * bit composes both: kTransposed false = congruous operand
 * (contract-contiguous, the only cp.async-able shape) into the canonical
 * [rows][kK] tile; true = crosswise 16-bit operand (row-contiguous 16B
 * runs) into the transposed [kK][rows] tile that ldmatrix.trans turns at
 * fragment time (b16-only; 8-bit crosswise keeps the LDG+PRMT staging).
 * Under trans staging the line axis is the contract dim, so the
 * predication axes trade places. The staged tile is the TRANS layout:
 * chunks swizzled by the k-row bits so the 8 k-rows of one ldmatrix.trans
 * matrix land on distinct chunks; only the low 3 row bits join the XOR
 * (LDSM gives 8 rows per matrix), so tiles wider than 8 chunks leave the
 * upper bits unswizzled. The tile arrives as a Tensor over the staged
 * layout (the swizzled address is its operator()). kInterior drops all
 * predication — valid only for a fully interior CTA (whole lines,
 * 16B-aligned base|ld, k_base + kK <= contract).
 */
template <typename SmemLayout,
          typename ElemT,
          int kThreads,
          bool kTransposed = false,
          bool kInterior = false>
DEVICE_FORCEINLINE void load_operand_tile(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                                          const ElemT* __restrict__ operand,
                                          int64_t rows,
                                          int64_t contract,
                                          int64_t ld,
                                          int tid,
                                          int64_t k_base,
                                          int64_t block_row) {
    static_assert(!kTransposed || sizeof(ElemT) == 2, "trans staging is 16-bit only");
    constexpr int kChunkElems = 16 / sizeof(ElemT);
    constexpr int kTileLines = SmemLayout::kRows; // lines staged per tile
    constexpr int kChunks = SmemLayout::kChunks;  // chunks per line
    constexpr int kTotalChunks = kTileLines * kChunks;
    /*
     * The bus may be under-subscribed (a 1-byte operand halves the chunks
     * per line; a 16-warp small CTA against a 1-byte B, or any kK=32
     * 1-byte side): threads take one chunk each, the rest stage nothing.
     * What cannot relax is the aligned run: the XOR chunk stepping
     * (dst ^ (j << 4)) is the swizzle of c0c + j only because a thread's
     * chunks are one aligned power-of-two run inside the line.
     */
    static_assert(kTotalChunks % kThreads == 0 || kTotalChunks < kThreads,
                  "tile chunks must divide the threads or under-subscribe the bus");
    constexpr int kCpt = // chunks per thread
        kTotalChunks < kThreads ? 1 : kTotalChunks / kThreads;
    static_assert(kCpt > 0 && (kCpt & (kCpt - 1)) == 0,
                  "XOR chunk stepping needs a power-of-two chunks-per-thread");
    /*
     * A thread's run must stay inside ONE line: a run wider than the line
     * sends kCpr to zero — tid/0 garbage and a wedged cp.async. Fully
     * subscribed this says the staged extent must not exceed the thread
     * count; the under-subscribed arm keeps kCpt 1 and cannot violate it.
     */
    static_assert(kCpt <= kChunks, "a thread's 16B run must fit one staged line: the staged "
                                   "extent cannot exceed the thread count");
    constexpr int kCpr = kChunks / kCpt; // chunks per line slice
    const int r = tid / kCpr;            // line within the tile
    const int c0 = (tid % kCpr) * kCpt * kChunkElems;
    if (r >= kTileLines)
        return; // no chunks for this thread on this bus
    /*
     * The mirror is three axes: the tile line sources from block_row
     * (canonical) or k_base (transposed) rows; the 16B run starts at
     * k_base (canonical) or block_row (transposed); and each axis is cut
     * by the extent that is NOT the one the run walks — the line's own
     * extent, then the other (non-contract vs contract).
     */
    const int64_t line0 = kTransposed ? k_base : block_row;
    const int64_t run0 = kTransposed ? block_row : k_base;
    const int64_t line_ext = kTransposed ? contract : rows;
    const int64_t run_ext = kTransposed ? rows : contract;
    if constexpr (kInterior) {
        const char* src = reinterpret_cast<const char*>(operand + (line0 + r) * ld + run0 + c0);
        const uintptr_t dst = reinterpret_cast<uintptr_t>(tile(r, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(reinterpret_cast<ElemT*>(dst ^ (j << 4)), src + j * 16);
    } else {
        const int64_t line = line0 + r;
        const bool line_ok = line < line_ext;
        /*
         * line0, run0, c0 and every j step are multiples of 16, so all
         * chunks share the run's alignment verdict (verdicts only differ
         * ACROSS lines, when ld is not 16B — see the scalar fallback).
         */
        const auto* src = operand + line * ld + run0 + c0;
        const bool chunk_aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
        const uintptr_t dst = reinterpret_cast<uintptr_t>(tile(r, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j) {
            const int c = j * kChunkElems;
            const int64_t col = run0 + c0 + c;
            ElemT* dstj = reinterpret_cast<ElemT*>(dst ^ (unsigned)(j << 4));
            if (chunk_aligned) {
                /*
                 * CUTLASS-style zero-fill predication: one cp.async whose
                 * runtime src-size loads the valid prefix (whole chunk,
                 * the extent tail cut or nothing for an OOB line) and the
                 * hardware zero-fills the remainder.
                 */
                const int64_t room = line_ok ? run_ext - col : 0;
                const int bytes = room >= kChunkElems ? 16
                                  : room > 0          ? (int)(room * (int64_t)sizeof(ElemT))
                                                      : 0;
                astrai::cp_async_16(dstj, src + c, bytes);
            } else {
                // Misaligned base only: element-granular fallback.
#pragma unroll
                for (int i = 0; i < kChunkElems; ++i)
                    dstj[i] = line_ok && col + i < run_ext ? src[c + i] : ElemT(0.0f);
            }
        }
    }
}

/*
 * Loop-carried prefetch state for one congruous-or-trans operand ring: the
 * per-thread (r, c0) swizzled stage destination and global source pointer
 * carried across k-tiles, so each prefetch chunk is one LDGSTS straight
 * from registers; geometry rides the operand's Ring tensor. kTrans selects
 * the crosswise 16-bit source geometry (rows are k lines: per-tile advance
 * kK*ld). kAsync false (synchronous 8-bit crosswise) collapses to no-ops.
 */
template <bool kAsync, typename RingT, int kThreads, bool kTrans = false> struct PrefetchCarry {
    using ElemT = typename RingT::Elem;               // Ring = Tensor<PtrEngine, RingLayout>
    using SmemLayout = typename RingT::Layout::Stage; // per-stage layout
    static constexpr int kChunkElems = 16 / sizeof(ElemT);
    /*
     * Same bus rule as load_operand_tile: an under-subscribed bus leaves
     * the surplus threads inactive rather than illegal.
     */
    static constexpr int kTotalChunks = SmemLayout::kRows * SmemLayout::kChunks;
    static_assert(kTotalChunks % kThreads == 0 || kTotalChunks < kThreads,
                  "tile chunks must divide the threads or under-subscribe the bus");
    static constexpr int kCpt = kTotalChunks < kThreads ? 1 : kTotalChunks / kThreads;
    static_assert(kCpt > 0 && (kCpt & (kCpt - 1)) == 0,
                  "XOR chunk stepping needs a power-of-two chunks-per-thread");
    static constexpr int kCpr = SmemLayout::kChunks / kCpt;
    // Carried state is in BYTES: both pointers advance by the stage stride.
    static constexpr int kK = kTrans ? SmemLayout::kRows : SmemLayout::kChunks * kChunkElems;
    static constexpr unsigned kKBytes = (unsigned)kK * sizeof(ElemT);
    unsigned wr = 0;           // current stage's swizzled destination offset
    unsigned wr0 = 0;          // slot-0 wrap base
    unsigned wrEnd = 0;        // one-past-the-ring sentinel
    const char* src = nullptr; // current tile's global source bytes
    int64_t srcStep = 0;       // per-tile source advance (bytes)
    bool active = true;        // false: bus under-subscribed, no chunks

    DEVICE_FORCEINLINE PrefetchCarry(const RingT& ring,
                                     const ElemT* operand,
                                     int64_t ld,
                                     int64_t blockRow,
                                     int tid,
                                     int firstTile) {
        if constexpr (kAsync) {
            const int rRaw = tid / kCpr;
            active = rRaw < SmemLayout::kRows;
            /*
             * An inactive thread's (r, c0) maps to no staged chunk; the carried
             * offsets are computed at a clamped r and never dereferenced.
             */
            const int r = active ? rRaw : 0;
            const int c0 = (tid % kCpr) * kCpt * kChunkElems;
            const ElemT* slot0 = astrai::stage_of(ring, firstTile).engine.ptr;
            const unsigned laneOff = static_cast<unsigned>(
                (const char*)astrai::stage_of(ring, firstTile)(r, c0) - (const char*)slot0);
            const unsigned base = __cvta_generic_to_shared(ring.engine.ptr) + laneOff;
            wr = base + (unsigned)((int64_t)(firstTile % RingT::Layout::kSlots) *
                                   RingT::Layout::kStageBytes);
            wr0 = base;
            wrEnd = base + (unsigned)RingT::Layout::kTotalBytes;
            if constexpr (kTrans) {
                src = reinterpret_cast<const char*>(operand + ((int64_t)firstTile * kK + r) * ld +
                                                    blockRow + c0);
                srcStep = (int64_t)kK * ld * sizeof(ElemT); // k advances rows
            } else {
                src = reinterpret_cast<const char*>(operand + (blockRow + r) * ld + c0) +
                      (int64_t)firstTile * kKBytes;
                srcStep = kKBytes; // k advances the contiguous columns
            }
        }
    }

    /*
     * Emit this thread's chunks for the current tile; pf false (loop tail)
     * zero-fills into the slot compute(i-1) already released. An inactive
     * thread owns no chunks, so it emits nothing at all.
     */
    DEVICE_FORCEINLINE void emit(bool pf) const {
        if constexpr (kAsync) {
            if (!active)
                return;
#pragma unroll
            for (int j = 0; j < kCpt; ++j)
                astrai::cp_async_16(wr ^ (unsigned)(j << 4), src + j * 16, pf);
        }
    }

    DEVICE_FORCEINLINE void advance() {
        if constexpr (kAsync) {
            wr += (unsigned)RingT::Layout::kStageBytes;
            if (wr == wrEnd)
                wr = wr0;
            src += srcStep;
        }
    }
};

} // namespace gemm
} // namespace astrai
