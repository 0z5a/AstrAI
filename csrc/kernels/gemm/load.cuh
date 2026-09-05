#pragma once
// Operand loaders: swizzled shared-memory staging for congruous operands
// (cp.async, predicated and interior variants, plus the loop-carried
// prefetch state) and the direct LDG+PRMT path for crosswise operands.
// The staging invariants and the swizzle derivation live in
// docs/developer/cuda_kernels.md.

#include "common/cp_async.cuh"
#include "gemm/common.h"
#include "policy.cuh"

namespace astrai {
namespace gemm {

// log2 of a compile-time power of two (for the swizzle shifts).
template <int N, int Acc = 0>
struct log2_const : log2_const<(N >> 1), Acc + 1> {};
template <int Acc>
struct log2_const<1, Acc> {
    static constexpr int value = Acc;
};

// Swizzled address inside a flat [rows * K] staging tile: the 16B chunk
// index is XORed with the row bits at [3, 3+log2(kChunks)) so a warp's
// ldmatrix fragment load (8 consecutive rows x 16B) hits all 32 banks
// exactly once; chunks stay contiguous, so cp.async staging is unaffected.
// All chunk math is in BYTES (16B granularity); element counts enter only
// through kChunkElems = 16 / sizeof(T).
template <int K, typename ElemT>
__device__ __forceinline__ ElemT* tile_at(ElemT* tile, int row, int col) {
    constexpr int kChunkElems = 16 / sizeof(ElemT);  // elems per 16B chunk
    constexpr int kChunks = K / kChunkElems;      // 16B chunks per row
    static_assert(kChunks >= 1 && (kChunks & (kChunks - 1)) == 0,
                  "swizzle needs a power-of-two 16B-chunk count");
    constexpr int kShift = 3 - log2_const<kChunks>::value;
    constexpr int kChunkShift = log2_const<kChunkElems>::value;
    return tile + row * K +
           ((((col >> kChunkShift) ^ ((row >> kShift) & (kChunks - 1)))
                 << kChunkShift) +
            (col & (kChunkElems - 1)));
}

// Stage-load a CONGRUOUS operand (contract-contiguous storage — the only
// cp.async-able shape) into the flat [rows * K] swizzled tile. kInterior
// drops all predication: valid only for a fully interior CTA (whole rows,
// 16B-aligned base|ld, k_base + K <= contract); the address math then folds
// to one immediate XOR per chunk (see the design notes). Crosswise operands
// go through load_crosswise_direct instead.
template <typename ElemT, int K, int RowsTile, int kThreads,
          bool kInterior = false>
__device__ __forceinline__ void
load_operand_tile(ElemT* tile, const ElemT* __restrict__ operand, int64_t rows,
                  int64_t contract, int64_t ld, int tid, int64_t k_base,
                  int64_t block_row) {
    constexpr int kChunkElems = 16 / sizeof(ElemT);
    constexpr int kChunks = K / kChunkElems;
    static_assert(RowsTile * kChunks % kThreads == 0,
                  "tile chunks must divide evenly across threads");
    constexpr int kCpt = RowsTile * kChunks / kThreads;  // chunks per thread
    constexpr int kCpr = kChunks / kCpt;  // chunks per row slice
    const int r = tid / kCpr;
    const int c0 = (tid % kCpr) * kCpt * kChunkElems;
    if constexpr (kInterior) {
        const char* src = reinterpret_cast<const char*>(
            operand + (block_row + r) * ld + k_base + c0);
        const uintptr_t dst =
            reinterpret_cast<uintptr_t>(tile_at<K>(tile, r, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(reinterpret_cast<ElemT*>(dst ^ (j << 4)),
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
            const int c = c0 + j * kChunkElems;
            ElemT* dst = tile_at<K>(tile, r, c);
            if (row_ok && chunk_aligned &&
                k_base + c + kChunkElems - 1 < contract) {
                astrai::cp_async_16(dst, src + c);
            } else {
                // Tail chunk / misaligned base / OOB row: scalar fill.
#pragma unroll
                for (int i = 0; i < kChunkElems; ++i)
                    dst[i] =
                        row_ok && k_base + c + i < contract ? src[c + i] : ElemT(0.0f);
            }
        }
    }
}

// Loop-carried prefetch state for one congruous operand ring: per-thread
// (r, c0) mapping with the swizzled stage destination and global source
// pointer carried across k-tiles, so each prefetch chunk is one LDGSTS
// issued straight from registers. The guard is a property of the operand's
// layout, so it lives in the type: the false specialization (crosswise
// operand) is an empty no-op.
template <bool kAsync, typename ElemT, int kK, int kRowsTile, int kThreads>
struct PrefetchCarry;

template <typename ElemT, int kK, int kRowsTile, int kThreads>
struct PrefetchCarry<true, ElemT, kK, kRowsTile, kThreads> {
    static constexpr int kChunkElems = 16 / sizeof(ElemT);
    static constexpr int kCpt = kRowsTile * (kK / kChunkElems) / kThreads;
    static constexpr int kCpr = (kK / kChunkElems) / kCpt;
    // All carried state is in BYTES: the smem write ring and the global
    // source pointer both advance by the byte-sized stage stride.
    static constexpr unsigned kKBytes = (unsigned)kK * sizeof(ElemT);
    unsigned wr = 0;    // current stage's swizzled destination offset
    unsigned wr0 = 0;   // slot-0 wrap base
    unsigned wrEnd = 0; // one-past-the-ring sentinel
    const char* src = nullptr;  // current tile's global source bytes

    __device__ __forceinline__ PrefetchCarry(
        const ElemT* ring, int ringSlots, int stageBytes, const ElemT* operand,
        int64_t ld, int64_t blockRow, int tid, int firstTile) {
        const int r = tid / kCpr;
        const int c0 = (tid % kCpr) * kCpt * kChunkElems;
        const int stageElems = stageBytes / (int)sizeof(ElemT);
        const ElemT* slot0 =
            ring + (int64_t)(firstTile % ringSlots) * stageElems;
        const unsigned laneOff = static_cast<unsigned>(
            (const char*)tile_at<kK>(slot0, r, c0) - (const char*)slot0);
        const unsigned base = __cvta_generic_to_shared(ring) + laneOff;
        wr = base + (unsigned)((int64_t)(firstTile % ringSlots) * stageBytes);
        wr0 = base;
        wrEnd = base + (unsigned)((int64_t)ringSlots * stageBytes);
        src = reinterpret_cast<const char*>(
                  operand + (blockRow + r) * ld + c0) +
              (int64_t)firstTile * kKBytes;
    }

    // Emit this thread's chunks for the current tile; pf false (loop tail)
    // zero-fills into the slot compute(i-1) already released.
    __device__ __forceinline__ void emit(bool pf) const {
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(wr ^ (unsigned)(j << 4), src + j * 16, pf);
    }

    __device__ __forceinline__ void advance(int stageBytes) {
        wr += (unsigned)stageBytes;
        if (wr == wrEnd) wr = wr0;
        src += kKBytes;
    }
};

template <typename ElemT, int kK, int kRowsTile, int kThreads>
struct PrefetchCarry<false, ElemT, kK, kRowsTile, kThreads> {
    __device__ __forceinline__ PrefetchCarry(
        const ElemT*, int, int, const ElemT*, int64_t, int64_t, int, int) {}
    __device__ __forceinline__ void emit(bool) const {}
    __device__ __forceinline__ void advance(int) {}
};

// Direct (synchronous) crosswise load into a canonical rotating stage:
// LDG.128 x4 (4 consecutive contract bytes x 16 rows) + in-register PRMT
// transpose + 16 STS.32. Crosswise operands cannot cp.async into the
// canonical tile (a 16B global run holds one contract byte for each of 16
// rows), so they take this path; a staged smem->smem variant measured
// 15-20% slower and was removed (see git history).
template <typename ElemT, int K, int RowsTile, int kThreads>
__device__ __forceinline__ void
load_crosswise_direct(ElemT* tile, const ElemT* __restrict__ operand, int64_t rows,
                      int64_t contract, int64_t ld, int tid, int64_t k_base,
                      int64_t block_row) {
    // The PRMT byte-perm transpose gathers 4 contract bytes per 32-bit word,
    // so this path is 1-byte-element only. 2-byte operands must arrive in
    // congruous storage (layout tags catch the mismatch at instantiation).
    static_assert(sizeof(ElemT) == 1,
                  "crosswise LDG+PRMT staging requires 1-byte elements");
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
                        *tile_at<K>(tile, rg * 16 + i, col) = ElemT(0.0f);
                    continue;
                }
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int64_t r_idx = r0 + i;
                    *tile_at<K>(tile, rg * 16 + i, col) =
                        r_idx < rows
                            ? operand[(k_base + col) * ld + r_idx]
                            : ElemT(0.0f);
                }
            }
        }
    }
}

}  // namespace gemm
}  // namespace astrai
