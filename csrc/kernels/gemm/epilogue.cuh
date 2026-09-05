#pragma once
// Collective epilogue: fused bias, the bf16 scatter of the fp32 accumulators
// through the reclaimed operand shared memory, and the coalesced copy-out.

#include "gemm/common.h"
#include "policy.cuh"

namespace astrai {
namespace gemm {

template <typename Policy>
struct GemmCollectiveEpilogue {
    using Traits = typename Policy::Traits;
    static constexpr bool kStreamOut = Policy::kStreamOut;
    static constexpr int kBlockM = Traits::kBlockM;
    static constexpr int kBlockN = Traits::kBlockN;
    static constexpr int kMt = Traits::kWarpM / 16;
    static constexpr int kNt = Traits::kWarpN / 8;

    __nv_bfloat16* const tile_out;
    const float output_scale;
    const __nv_bfloat16* const bias;
    const int64_t m, n, out_ld;
    // Output orientation from the policy tag (CUTLASS LayoutC): the NN
    // swap computes E = B^T A^T, instantiated with LayoutOut = ColMajor.
    static constexpr bool t_out =
        !std::is_same_v<typename Policy::LayoutTagOut, RowMajor>;
    const int row_elems, row_chunks;
    const int warp_m, warp_n, group, thread_in_group;
    const int64_t block_m, block_n;

    __device__ GemmCollectiveEpilogue(char* smem, const GemmParams& p,
                                     int64_t block_m, int64_t block_n, int tid)
        : tile_out(reinterpret_cast<__nv_bfloat16*>(smem)),
          output_scale(Traits::kNeedsDequant ? *p.scale : 1.0f),
          bias(reinterpret_cast<const __nv_bfloat16*>(p.bias_ptr)),
          m(p.m), n(p.n), out_ld(p.out_ld),
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
        // the swapped orientation the bias indexes D-cols = the kernel's rows.
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
            const int64_t row_stride = out_ld;
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

}  // namespace gemm
}  // namespace astrai
