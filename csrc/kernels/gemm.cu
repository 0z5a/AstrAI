// BF16 decode-time GEMM primitive for linear layers: one entry point whose
// internal path is sized by the decode batch M.
//
//   M in [1, 8]  — one CTA per weight row, the M rows held in registers
//                  (any K).
//   M in (8, 32] — BM=16 tensor-core tiles; shape-driven configs (see
//                  the dispatch table at the entry point) cover every
//                  production shape (K % 8 == 0 and 16-byte-aligned
//                  tensors).
//
// Both paths take row-major [N, K] weights, accumulate in FP32, fuse an
// optional bias, and are inference-only.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

// Route kernel-launch failures through the torch error check instead of
// common/launch.cuh's print+abort default. Must precede the kernel code.
#define ASTRAI_LAUNCH_FAIL(err, what) C10_CUDA_CHECK(err)

#include "common/cp_async.cuh"
#include "common/launch.cuh"
#include "common/mma.cuh"

namespace {

constexpr int kHalfCtaThreads = 128;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int Rows, int Threads>
__global__ void skinny_gemm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int n,
    int k
) {
    const int output_index = blockIdx.x;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;

    float sums[Rows] = {};
    __shared__ float warp_sums[Rows][Threads / kWarpSize];
    // Weight row: scalar head/tail around a 16-byte-aligned uint4 middle so
    // any K is accepted while keeping 128-bit weight loads, which dominate
    // bandwidth on decode shapes. x pairs with scalar loads: it is a tiny
    // L1/L2-resident matrix, consecutive threads still touch contiguous
    // addresses, and no per-row alignment case analysis is needed.
    const __nv_bfloat16* __restrict__ wrow =
        weight + static_cast<int64_t>(output_index) * k;
    const unsigned whead_raw =
        ((16u - (reinterpret_cast<uintptr_t>(wrow) & 15u)) & 15u) >> 1;
    const int whead = static_cast<int>(min(whead_raw, static_cast<unsigned>(k)));
    const int wvecs = (k - whead) / 8;
    const int wtail_start = whead + wvecs * 8;
    const uint4* __restrict__ w4 = reinterpret_cast<const uint4*>(wrow + whead);

    // x chunks pair element-for-element with the aligned weight middle:
    // the uint4 view is rooted at ``x + whead`` (16-byte aligned by the
    // branch guard), and each row strides by ``k / 8`` vectors because its
    // first middle element sits ``whead`` scalars past ``row * k``. When
    // K % 8 == 0 and the weight row is already aligned (whead == 0, the
    // production case) this reduces to one pure uint4 loop with an empty
    // head/tail. Otherwise per-row uint4 loads are not 16-byte addressable,
    // and scalar x pairing keeps the kernel correct for any K while the
    // weight stream stays vectorized.
    if (k % 8 == 0 &&
        ((reinterpret_cast<uintptr_t>(x) + 2u * static_cast<unsigned>(whead)) & 15u) == 0u) {
        const auto* x4 = reinterpret_cast<const uint4*>(x + whead);
        for (int v = threadIdx.x; v < wvecs; v += blockDim.x) {
            const uint4 wv_raw = w4[v];
            const auto* wv =
                reinterpret_cast<const __nv_bfloat162*>(&wv_raw);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const uint4 xv_raw =
                    x4[(static_cast<int64_t>(row) * (k / 8)) + v];
                const auto* xv =
                    reinterpret_cast<const __nv_bfloat162*>(&xv_raw);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sums[row] = fmaf(
                        __bfloat162float(__low2bfloat16(xv[p])),
                        __bfloat162float(__low2bfloat16(wv[p])),
                        sums[row]
                    );
                    sums[row] = fmaf(
                        __bfloat162float(__high2bfloat16(xv[p])),
                        __bfloat162float(__high2bfloat16(wv[p])),
                        sums[row]
                    );
                }
            }
        }
    } else {
        for (int v = threadIdx.x; v < wvecs; v += blockDim.x) {
            const uint4 wv_raw = w4[v];
            const __nv_bfloat16* wv_s =
                reinterpret_cast<const __nv_bfloat16*>(&wv_raw);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const __nv_bfloat16* xv =
                    x + static_cast<int64_t>(row) * k + whead + 8 * v;
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    sums[row] = fmaf(
                        __bfloat162float(xv[s]),
                        __bfloat162float(wv_s[s]),
                        sums[row]
                    );
                }
            }
        }
    }
    // Head and tail remainders: plain scalar pairing, at most 14 elements.
    for (int i = threadIdx.x; i < whead; i += blockDim.x) {
        const float wv = __bfloat162float(wrow[i]);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            sums[row] = fmaf(
                __bfloat162float(x[static_cast<int64_t>(row) * k + i]),
                wv,
                sums[row]
            );
        }
    }
    for (int i = wtail_start + threadIdx.x; i < k; i += blockDim.x) {
        const float wv = __bfloat162float(wrow[i]);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            sums[row] = fmaf(
                __bfloat162float(x[static_cast<int64_t>(row) * k + i]),
                wv,
                sums[row]
            );
        }
    }

#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        sums[row] = warp_sum(sums[row]);
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            warp_sums[row][warp] = sums[row];
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float sum =
                lane < (Threads / kWarpSize) ? warp_sums[row][lane] : 0.0f;
            sum = warp_sum(sum);
            if (lane == 0) {
                if (bias != nullptr) {
                    sum += __bfloat162float(bias[output_index]);
                }
                output[row * n + output_index] = __float2bfloat16_rn(sum);
            }
        }
    }
}

template <int Rows, int Threads>
void launch_skinny_gemm(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_bfloat16* output,
    int n,
    int k,
    cudaStream_t stream
) {
    // Split-K was tried and rejected for the small-N shapes (GQA kv
    // projections): their ~48K uint4 loads already saturate thread-level
    // parallelism one load deep, so they sit on the launch+HBM latency
    // floor, and the fence+atomic+last-CTA partial round trip adds ~0.6us
    // of fixed sync cost (measured -27% at M=1, -113% at M=8 on L20).
    // The real fix for those shapes is fusing the QKV projections so the
    // tiny kv rows stop launching as standalone kernels at all.
    //
    // Decode is HBM weight-streaming bound. Every shape launches with a
    // single 128-thread CTA size: measured on L20 (sm_89) with L2-thrashing
    // weight rotation, flat 128t is within ~1% of a per-shape tuned
    // 128/256/512 mix at M=1 and M=8 and gives up at most ~2% at M=2-4
    // (512t down_proj, 256t gate/up/lm_head cells), while dodging the
    // Rows>=7 register cliff on the 307MB lm_head (+21% DRAM throughput
    // vs 256t at M=8). Re-measured
    // after the table refactor: 256t on wide-N gate/up wins only ~2.8% at
    // M=2-4 and ties at M=1/6, inside run-to-run drift. Simplicity keeps
    // winning over the last ~2%.
    skinny_gemm_kernel<Rows, Threads>
        <<<n, Threads, 0, stream>>>(
            x, weight, bias, output, n, k
        );
    ASTRAI_LAUNCH_CHECK();
}

// Compile-time dispatch tables: replace a hand-written switch over M with
// function-pointer tables indexed by M-1. Adding a Rows variant means adding
// one table entry, not a new case block at the call site.
using SkinnyGemmFn = void (*)(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, cudaStream_t
);

constexpr SkinnyGemmFn kSkinnyGemm[8] = {
    &launch_skinny_gemm<1, kHalfCtaThreads>,
    &launch_skinny_gemm<2, kHalfCtaThreads>,
    &launch_skinny_gemm<3, kHalfCtaThreads>,
    &launch_skinny_gemm<4, kHalfCtaThreads>,
    &launch_skinny_gemm<5, kHalfCtaThreads>,
    &launch_skinny_gemm<6, kHalfCtaThreads>,
    &launch_skinny_gemm<7, kHalfCtaThreads>,
    &launch_skinny_gemm<8, kHalfCtaThreads>,
};

// ---------------------------------------------------------------------------
// M > 8 path: small-M (M <= 64) tiled GEMM.
//
// F.linear for decode batches in (8, 64]: cuBLAS tiles the small M as a
// single tile row, which starves the grid (measured on L20, sm_89: 24-54
// CTAs on 92 SMs, tensor pipe 24-42%, DRAM <= 72%). This kernel keeps the
// M rows in one CTA tile and fills the SMs along N and K instead.
//
// CUTLASS-style configuration: the kernel is parameterized by template
// parameters — CTA tile (BM x BN x BK), pipeline depth, thread count —
// with the warp layout derived inside (kMt M fragments, kNt n16 tiles per
// warp). Four families are instantiated (see the dispatch table at the
// entry point): the default (BN=64, BK=64, 3 stages) and its BM=32 wide-N
// variant, plus narrow-N deep-K rings (BK=256/128, BN=32, 64 threads); a
// new shape is an instantiation, not a rewrite.
//
// Operand staging reuses the FP8 GEMM's scheme (see fp8/gemm/*.cuh and
// docs/developer/cuda_kernels.md): 16B chunks XOR-swizzled with row & 7,
// one barrier per k-tile, and a kStages+1 ring whose prefetch for tile
// i+kStages lands in the slot tile i-1 released — no post-compute barrier.
// ---------------------------------------------------------------------------

using bf16 = __nv_bfloat16;

// Logical (row, byte column) -> byte offset in one flat [rows * BK*2B]
// staging tile. The 16B chunk index is XORed with row & (chunks - 1) so an
// ldmatrix fragment load (8 consecutive rows x 16B) hits all 32 banks once.
template <int BK>
__device__ __forceinline__ int tile_off(int row, int byte_col) {
    constexpr int kRowBytes = BK * 2;
    constexpr int kChunks = kRowBytes / 16;
    static_assert(
        (kChunks & (kChunks - 1)) == 0, "swizzle needs a power-of-two chunk count"
    );
    return row * kRowBytes +
           (((byte_col >> 4) ^ (row & (kChunks - 1))) << 4) + (byte_col & 15);
}

// Predicated staging of one [RowsTile x BK] operand slice from a
// row-major [total_rows, K] tensor into its swizzled ring slot. Chunks past
// the row count or past K zero-fill (the wrapper guarantees K % 8 == 0 and
// 16B-aligned rows, so a misaligned *valid* chunk cannot occur).
template <int RowsTile, int BK, int kThreads>
__device__ __forceinline__ void stage_tile(
    char* slot, const bf16* __restrict__ src, int total_rows, int64_t k,
    int row0, int tid, int kt
) {
    constexpr int kRowBytes = BK * 2;
    constexpr int kChunks = kRowBytes / 16;
#pragma unroll
    for (int c = tid; c < RowsTile * kChunks; c += kThreads) {
        const int r = c / kChunks;
        const int c8 = c % kChunks;
        const int64_t kbase = (int64_t)kt * BK;
        const bool ok = row0 + r < total_rows && kbase + c8 * 8 + 8 <= k;
        char* dst = slot + tile_off<BK>(r, c8 * 16);
        if (ok) {
            astrai::cp_async_16(
                reinterpret_cast<bf16*>(dst),
                src + (int64_t)(row0 + r) * k + kbase + c8 * 8
            );
        } else {
            unsigned* w = reinterpret_cast<unsigned*>(dst);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                w[i] = 0u;
        }
    }
}

template <int BM, int BN, int BK, int kStages, int kThreads>
__global__ __launch_bounds__(kThreads, 1) void tiled_gemm_kernel(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int m,
    int n,
    int k
) {
    constexpr int kMt = BM / 16;              // m16 fragments
    constexpr int kWarps = kThreads / 32;
    constexpr int kNt = BN / (kWarps * 16);   // n16 tiles per warp
    constexpr int kRowBytes = BK * 2;
    constexpr int kRing = kStages + 1;        // ring slots per operand
    constexpr int kSegs = BK / 16;            // m16k16 segments per tile
    constexpr int kSegXor = 32;               // bytes: +2 chunks per segment
    static_assert(BM % 16 == 0, "BM must be a multiple of 16");
    static_assert(BN % (kWarps * 16) == 0, "warps must tile BN in n16 units");

    extern __shared__ __align__(16) char smem[];
    char* const a_ring = smem;
    char* const b_ring = smem + kRing * BM * kRowBytes;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    // Standard 2D grid mapping: grid.x covers M tiles, grid.y covers N
    // tiles. The grid fills the SMs along N; every block walks the whole
    // K range in a single pass.
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int total_tiles = (k + BK - 1) / BK;

    auto a_slot = [&](int t) {
        return a_ring + (t % kRing) * BM * kRowBytes;
    };
    auto b_slot = [&](int t) {
        return b_ring + (t % kRing) * BN * kRowBytes;
    };

#pragma unroll
    for (int s = 0; s < kStages; ++s) {
        if (s < total_tiles) {
            stage_tile<BM, BK, kThreads>(
                a_slot(s), x, m, k, m0, tid, s
            );
            stage_tile<BN, BK, kThreads>(b_slot(s), w, n, k, n0, tid, s);
        }
        astrai::cp_async_commit_group();
    }

    // Per-lane ldmatrix fragment addresses (relative to each ring slot):
    //   A x4: lanes 0-7 mat0 (rows 0-7, chunk k-lo), 8-15 mat1 (rows 8-15,
    //   k-lo), 16-23 mat2 (rows 0-7, k-hi), 24-31 mat3 (rows 8-15, k-hi).
    //   B x4 pair: lanes 0-7/8-15 the first n8 tile's k-lo/k-hi chunks,
    //   16-23/24-31 the second n8 tile's. Warp w owns the n16 tiles at
    //   (w + j * kWarps) * 16 for j in [0, kNt).
    const int a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const unsigned a_off = tile_off<BK>(a_row, (lane >> 4) * 16);
    unsigned b_off[kNt];
#pragma unroll
    for (int j = 0; j < kNt; ++j) {
        const int b_row =
            (warp + j * kWarps) * 16 + (lane & 7) + (lane >> 4) * 8;
        b_off[j] = tile_off<BK>(b_row, ((lane >> 3) & 1) * 16);
    }
    const unsigned a_base0 = __cvta_generic_to_shared(a_ring) + a_off;
    unsigned b_base0[kNt];
#pragma unroll
    for (int j = 0; j < kNt; ++j)
        b_base0[j] = __cvta_generic_to_shared(b_ring) + b_off[j];

    float acc[kMt][kNt][2][4] = {};
    for (int i = 0; i < total_tiles; ++i) {
        astrai::cp_async_wait_group<kStages - 1>();
        __syncthreads();
        const unsigned a_base =
            a_base0 + (unsigned)((i % kRing) * BM * kRowBytes);
        unsigned b_base[kNt];
#pragma unroll
        for (int j = 0; j < kNt; ++j)
            b_base[j] = b_base0[j] + (unsigned)((i % kRing) * BN * kRowBytes);
#pragma unroll
        for (int seg = 0; seg < kSegs; ++seg) {
            const unsigned a_seg = a_base ^ (unsigned)(seg * kSegXor);
            unsigned a4[kMt][4], b4[kNt][4];
#pragma unroll
            for (int mt = 0; mt < kMt; ++mt)
                astrai::ldmatrix_x4_lane(
                    a4[mt], a_seg + (unsigned)(mt * 16 * kRowBytes)
                );
#pragma unroll
            for (int j = 0; j < kNt; ++j)
                astrai::ldmatrix_x4_lane(
                    b4[j], (b_base[j] ^ (unsigned)(seg * kSegXor))
                );
#pragma unroll
            for (int mt = 0; mt < kMt; ++mt)
#pragma unroll
                for (int j = 0; j < kNt; ++j)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
                        astrai::mma_sync<bf16>(
                            acc[mt][j][nt], a4[mt], b4[j] + nt * 2,
                            acc[mt][j][nt]
                        );
        }
        // Prefetch tile i+kStages into the slot tile i-1 released. The
        // barrier at the top of the next iteration separates every
        // thread's reads of that slot (iteration i-1) from these writes.
        const int pf = i + kStages;
        if (pf < total_tiles) {
            stage_tile<BM, BK, kThreads>(
                a_slot(pf), x, m, k, m0, tid, pf
            );
            stage_tile<BN, BK, kThreads>(b_slot(pf), w, n, k, n0, tid, pf);
        }
        astrai::cp_async_commit_group();
    }

    const int row0 = lane >> 2;
    const int col0 = (lane & 3) * 2;
#pragma unroll
    for (int mt = 0; mt < kMt; ++mt) {
        if (m0 + mt * 16 + row0 >= m)
            continue;
        const int64_t orow = (int64_t)(m0 + mt * 16 + row0) * n;
#pragma unroll
        for (int j = 0; j < kNt; ++j) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int col =
                    n0 + (warp + j * kWarps) * 16 + col0 + nt * 8;
                if (col >= n)
                    continue;
                float2 v, v8;
                v.x = acc[mt][j][nt][0];
                v.y = acc[mt][j][nt][1];
                v8.x = acc[mt][j][nt][2];
                v8.y = acc[mt][j][nt][3];
                if (bias != nullptr) {
                    v.x += __bfloat162float(bias[col]);
                    v.y += __bfloat162float(bias[col + 1]);
                    v8.x += __bfloat162float(bias[col]);
                    v8.y += __bfloat162float(bias[col + 1]);
                }
                if (col + 1 < n) {
                    *reinterpret_cast<__nv_bfloat162*>(out + orow + col) =
                        __floats2bfloat162_rn(v.x, v.y);
                    if (m0 + mt * 16 + row0 + 8 < m)
                        *reinterpret_cast<__nv_bfloat162*>(
                            out + orow + 8 * n + col) =
                            __floats2bfloat162_rn(v8.x, v8.y);
                } else {
                    out[orow + col] = __float2bfloat16_rn(v.x);
                    if (m0 + mt * 16 + row0 + 8 < m)
                        out[orow + 8 * n + col] =
                            __float2bfloat16_rn(v8.x);
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int kStages, int kThreads>
void launch_tiled_gemm(
    const bf16* x,
    const bf16* w,
    const bf16* bias,
    bf16* out,
    int m,
    int n,
    int k,
    cudaStream_t stream
) {
    constexpr int kRing = kStages + 1;
    constexpr int smem = kRing * (BM + BN) * BK * 2;
    // 99KB is the sm_86/89 per-block opt-in ceiling; configs above 48KB
    // (long-K BK=128) run one CTA per SM and pay a one-time attribute opt-in.
    static_assert(smem <= 99 * 1024, "family must fit the sm_86/89 opt-in ceiling");
    if constexpr (smem > 48 * 1024) {
        static const bool opted_in = [] {
            ASTRAI_CUDA_CHECK(cudaFuncSetAttribute(
                tiled_gemm_kernel<BM, BN, BK, kStages, kThreads>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem
            ));
            return true;
        }();
        (void)opted_in;
    }
    dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN);
    tiled_gemm_kernel<BM, BN, BK, kStages, kThreads>
        <<<grid, kThreads, smem, stream>>>(
            x, w, bias, out, m, n, k
        );
    ASTRAI_LAUNCH_CHECK();
}

using TiledGemmFn = void (*)(
    const bf16*, const bf16*, const bf16*, bf16*, int, int, int,
    cudaStream_t
);

// CTA tile configuration pairing the shape parameters with their matched
// instantiation. Field order is canonical everywhere it appears — template
// arguments, this struct, the dispatch table — as BM, BN, BK, then
// pipeline depth (stages) and CTA size (threads). A new family is one
// row here plus one select branch.
struct TileConfig {
    int bm;
    int bn;
    int bk;
    int stages;
    int threads;
    TiledGemmFn launch;
};

// Shape -> tile config, measured on L20 with L2-thrashing weight rotation.
// The selector trades grid fill against K-loop serial latency:
//  - Wide N (n >= 4096): ceil(n/64) tiles already cover the SMs, the GEMM
//    is HBM-bound and the default config wins; BM widens to 32 at M>16 to
//    halve the re-staged weight stream.
//  - Narrow N: few N tiles leave the grid K-serial — widening the grid
//    does not help (measured: BN 64->32 ties, doubled m_tiles tie, kv at
//    4 blocks ties q/o at 24); fewer, deeper K chunks do. BK=256 with a
//    72KB two-stage ring is the winner while the grid fits one wave
//    (72KB smem means one CTA per SM); past one wave its 2-wave
//    quantization loses to BK=128's 36.9KB two-CTA ring.
inline TileConfig select_tile_config(int m, int n, int k) {
    if (n >= 4096) {
        if (m > 16) {
            return {32, 64, 64, 3, 128,
                    &launch_tiled_gemm<32, 64, 64, 3, 128>};
        }
        return {16, 64, 64, 3, 128, &launch_tiled_gemm<16, 64, 64, 3, 128>};
    }
    const int n_tiles = (n + 31) / 32;
    const int m_tiles = (m + 15) / 16;
    if (n_tiles * m_tiles <= 92) {  // one wave on the 92-SM L20
        return {16, 32, 256, 2, 64,  &launch_tiled_gemm<16, 32, 256, 2, 64>};
    }
    return {16, 32, 128, 2, 64, &launch_tiled_gemm<16, 32, 128, 2, 64>};
}

}  // namespace

// Single entry point: M in [1, 8] routes to the register-resident skinny
// GEMM kernel (any K), M in (8, 64] to the tiled kernel (K % 8 == 0 and
// 16-byte-aligned tensors, checked at the branch).
torch::Tensor bf16_gemm(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_object
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x and weight must be CUDA tensors");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must share device");
    TORCH_CHECK(
        x.scalar_type() == torch::kBFloat16 &&
            weight.scalar_type() == torch::kBFloat16,
        "x and weight must be bf16"
    );
    TORCH_CHECK(
        x.dim() == 1 || x.dim() == 2,
        "x must have shape [K] or [M, K]"
    );
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "x and weight must be contiguous");
    TORCH_CHECK(
        !x.requires_grad() && !weight.requires_grad(),
        "bf16_gemm is inference-only and does not support autograd"
    );

    const int64_t m = x.dim() == 1 ? 1 : x.size(0);
    const int64_t k = x.size(-1);
    const int64_t n = weight.size(0);
    TORCH_CHECK(weight.size(1) == k, "weight K must match x K");
    TORCH_CHECK(m >= 1 && m <= 64, "M must be in [1, 64]");
    TORCH_CHECK(k > 0 && n > 0, "N and K must be positive");
    TORCH_CHECK(
        k <= std::numeric_limits<int>::max() &&
            n <= std::numeric_limits<int>::max(),
        "N or K exceeds the CUDA launcher limit"
    );

    torch::Tensor bias;
    const __nv_bfloat16* bias_ptr = nullptr;
    if (!bias_object.is_none()) {
        bias = bias_object.cast<torch::Tensor>();
        TORCH_CHECK(bias.is_cuda() && bias.device() == x.device(), "bias must share the CUDA device");
        TORCH_CHECK(bias.scalar_type() == torch::kBFloat16, "bias must be bf16");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == n, "bias must have shape [N]");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(!bias.requires_grad(), "bf16_gemm bias does not support autograd");
        bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
    }

    const at::cuda::OptionalCUDAGuard guard(x.device());
    const auto* properties = at::cuda::getDeviceProperties(x.device().index());
    TORCH_CHECK(properties->major >= 8, "bf16_gemm requires compute capability 8.0+");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto output = x.dim() == 1 ? torch::empty({n}, x.options())
                               : torch::empty({m, n}, x.options());

    const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* weight_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
    auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    const int m_int = static_cast<int>(m);
    const int n_int = static_cast<int>(n);
    const int k_int = static_cast<int>(k);

    if (m <= 8) {
        kSkinnyGemm[m_int - 1](
            x_ptr, weight_ptr, bias_ptr, output_ptr, n_int, k_int,
            stream.stream()
        );
    } else {
        TORCH_CHECK(
            k % 8 == 0 &&
                (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0u &&
                (reinterpret_cast<uintptr_t>(weight.data_ptr()) & 15) == 0u,
            "M > 8 requires K to be a multiple of 8 and x/weight 16-byte aligned"
        );
        const TileConfig cfg = select_tile_config(m_int, n_int, k_int);
        cfg.launch(
            x_ptr, weight_ptr, bias_ptr, output_ptr, m_int, n_int, k_int,
            stream.stream()
        );
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "bf16_gemm",
        &bf16_gemm,
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        "M in [1, 64] BF16 GEMM with optional fused bias "
        "(register-resident skinny GEMM path for M <= 8, tensor-core tiles above)"
    );
}
