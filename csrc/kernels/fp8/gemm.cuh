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
    return static_cast<unsigned>(__nv_cvt_float2_to_fp8x2(
        make_float2(lo * inv, hi * inv), __NV_SATFINITE, kFmt));
}

template <FP8Format Fmt>
__global__ void fp8_quantize_kernel(FP8QuantizeParams p) {
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
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(x8)) & 15) ==
        0;
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
    if (p.ring_state && amax) {
        // Delayed-scaling ring finalization as a last-block epilogue (the
        // CUDA threadFenceReduction pattern): the fence + counter elect the
        // final block once every block's atomic_max above is visible; warp 0
        // folds the fresh amax into the window, reduces it and publishes the
        // next step's scale, then re-arms the counter for the next launch.
        // __fdiv_rn / ldexpf keep the scale bit-identical to the eager
        // (peak / fp8_max) / 2^margin fp32 chain despite --use_fast_math.
        __threadfence();
        __shared__ bool ring_last;
        if (threadIdx.x == 0)
            ring_last = atomicAdd(reinterpret_cast<int*>(p.ring_state +
                                                        p.ring_len + 1),
                                  1) == gridDim.x - 1;
        __syncthreads();
        if (ring_last && threadIdx.x < 32) {
            float* hist = p.ring_state;
            const int lane = threadIdx.x;
            float v = 0.0f;
            if (lane < p.ring_len) v = hist[lane];
            if (lane == p.ring_idx) {
                v = *amax;  // the global amax is final now
                hist[lane] = v;
            }
            // Windows longer than one warp (atypical) fold the tail.
            for (int i = lane + 32; i < p.ring_len; i += 32) {
                float h = hist[i];
                if (i == p.ring_idx) {
                    h = *amax;
                    hist[i] = h;
                }
                v = fmaxf(v, h);
            }
            const float peak = warp_reduce_max(v);
            if (lane == 0) {
                constexpr float kFmtMax =
                    Fmt == FP8Format::E5M2 ? 57344.0f : 448.0f;
                p.ring_state[p.ring_len] = fmaxf(
                    ldexpf(__fdiv_rn(peak, kFmtMax), -p.ring_margin), 1e-12f);
                __threadfence();
                // Re-arm the counter (0.0f bits == int32 0).
                p.ring_state[p.ring_len + 1] = 0.0f;
            }
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
    return tile + row * K +
           ((((col >> 4) ^ ((row >> kShift) & (kChunks - 1))) << 4) + (col & 15));
}

// Stage-load a CONGRUOUS operand (stored [rows][contract], contract-
// contiguous — the only cp.async-able shape for the canonical tile) into the
// flat [rows * K] shared tile via tile_at's swizzle. Crosswise operands go
// through stage_crosswise_tile + transpose_crosswise_tile instead.
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
            astrai::cp_async_16(dst, src + c, true);
        } else {
            // Tail chunk (or misaligned base): predicated scalar fill.
#pragma unroll
            for (int i = 0; i < 16; ++i)
                dst[i] =
                    row_ok && k_base + c + i < contract ? src[c + i] : T8(0.0f);
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
// pointer for ldmatrix. Valid for kK in {32, 64, 128} (the swizzle itself
// lives only in tile_at; this wrapper just converts the element address).
template <typename T8, int kK>
__device__ __forceinline__ unsigned frag_addr(const T8* tile, int row, int chunk) {
    static_assert(kK == 32 || kK == 64 || kK == 128,
                  "fragment swizzle offsets assume kK in {32, 64, 128}");
    return __cvta_generic_to_shared(tile_at<kK>(tile, row, chunk << 4));
}

// Crosswise operands (stored [contract][rows], rows-contiguous) cannot be
// cp.async'd into the canonical [rows][contract] tile — a 16B global run
// holds one contract byte for each of 16 rows. They stage K-major instead
// (byte (p, r) at p*RowsTile + r), where the very same runs land contiguously
// and cp.async applies unchanged; a per-tile smem->smem transpose (below)
// then produces the canonical swizzled tile the MMA fragments read. This
// keeps the whole global→shared path asynchronous — the synchronous
// LDG+byte-scatter staging this replaces left the kernel long-scoreboard
// bound (ncu: 4.6 stalled loads per issue vs 0.4 on the congruous path).
template <typename T8, int K, int RowsTile, int kThreads>
__device__ __forceinline__ void
stage_crosswise_tile(T8* staging, const T8* __restrict__ operand, int64_t rows,
                     int64_t contract, int64_t ld, int tid, int64_t k_base,
                     int64_t block_row) {
    constexpr int kRuns = K * RowsTile / 16;  // 16B runs per tile
    // r0 is a multiple of 16 and p*ld keeps 16B alignment whenever ld has it,
    // so one uniform verdict covers every run.
    const bool run_aligned =
        ((reinterpret_cast<uintptr_t>(operand) | ld) & 15) == 0;
    for (int run = tid; run < kRuns; run += kThreads) {
        const int pl = run % K;  // local contract byte (column of the run)
        const int rg = run / K;  // 16-row group
        const int64_t r0 = block_row + (int64_t)rg * 16;
        T8* dst = staging + pl * RowsTile + rg * 16;
        if (run_aligned && r0 + 15 < rows && k_base + pl < contract)
            astrai::cp_async_16(dst, operand + (k_base + pl) * ld + r0, true);
        else {
            // Row tail, contract tail or misaligned base: predicated fill.
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int64_t r = r0 + i;
                dst[i] = r < rows && k_base + pl < contract
                             ? operand[(k_base + pl) * ld + r]
                             : T8(0.0f);
            }
        }
    }
}

// Direct (synchronous) crosswise load into a canonical rotating stage:
// LDG.128 x4 (4 consecutive contract bytes x 16 rows) + in-register PRMT
// transpose + 16 STS.32. Used for crosswise operands whose global data is
// typically L2-resident (the A side of dW): the staging detour's extra
// shared-memory round trip costs more than the latency it hides there,
// while crosswise B operands (DRAM-streamed weights of dX) take the
// asynchronous stage_crosswise_tile path instead.
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

// K-major staging -> canonical [rows][kK] swizzled tile, one chunk at a time.
// Each chunk (indexed within a k_seg region of `quads_per_seg` quads) covers
// 4 consecutive contract bytes x 16 rows: four LDS.128 grab the staging runs,
// PRMT byte selects transpose them in registers, and sixteen STS.32 land the
// row quads through tile_at's swizzle — 4x fewer store instructions than a
// byte-granular scatter. Chunk-at-a-time lets the caller pool work across
// operands; the region restriction lets the main loop overlap one region's
// transpose with another region's MMAs (a whole-tile serial transpose put
// the crosswise GEMMs at 25% tensor utilization).
template <typename T8, int K, int RowsTile>
__device__ __forceinline__ void
transpose_crosswise_region(T8* tile, const T8* staging, int idx, int quad0) {
    constexpr int kGroups = RowsTile / 16;
    const int quad = quad0 + idx / kGroups;
    const int rg = idx % kGroups;
    // The four runs sit RowsTile bytes apart (one per contract byte of the
    // quad); each run is 16 contiguous staging bytes = 16 rows.
    const char* run0 = reinterpret_cast<const char*>(
        staging + quad * 4 * RowsTile + rg * 16);
    uint4 v[4];
#pragma unroll
    for (int s = 0; s < 4; ++s)
        v[s] = *reinterpret_cast<const uint4*>(run0 + s * RowsTile);
    const unsigned* bytes = reinterpret_cast<const unsigned*>(v);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        // word i = row r0+i's quad: byte i of each of the four runs
        // [v0.b(i), v1.b(i), v2.b(i), v3.b(i)]. Byte i of a uint4 lives in
        // its (i>>2)-th 32-bit register.
        const unsigned nib = i & 3;
        const unsigned sel = nib | ((nib + 4) << 4);
        const unsigned w01 =
            __byte_perm(bytes[0 + (i >> 2)], bytes[4 + (i >> 2)], sel);
        const unsigned w23 =
            __byte_perm(bytes[8 + (i >> 2)], bytes[12 + (i >> 2)], sel);
        *reinterpret_cast<unsigned*>(tile_at<K>(tile, rg * 16 + i, quad * 4)) =
            __byte_perm(w01, w23, 0x5410u);
    }
}

// Layout-aware shared-memory budget and occupancy hint. A congruous operand
// needs its kStages rotating canonical buffers; a staged-crosswise operand
// (crosswise B with kBStaged) needs kStages K-major staging buffers plus ONE
// canonical buffer (rewritten every tile by the in-kernel transpose); a
// direct-crosswise operand rotates kStages+1 canonical buffers so its load
// can run ahead of the compute phase (see the kernel's pipelining note).
// The 48KB static-smem watermark picks the resident-CTA hint for
// __launch_bounds__ (sm_89: 100KB smem per SM, so two CTAs fit while each
// stays within the static budget).
template <typename Traits, typename LayoutA, typename LayoutB, bool StagedB>
struct Fp8GemmSmem {
    // Crosswise = the stage-load's view: A's tag directly, B's transposed.
    // A-crosswise always loads direct (L2-typical activations); B-crosswise
    // stages only when its contract dim is long enough to stream DRAM.
    static constexpr bool kCrossA = std::is_same_v<LayoutA, ColMajor>;
    static constexpr bool kCrossB = std::is_same_v<LayoutB, RowMajor>;
    static constexpr bool kBStagePath = kCrossB && StagedB;
    static constexpr bool kDirectA = kCrossA;
    static constexpr bool kDirectB = kCrossB && !kBStagePath;
    static constexpr int kBytes =
        (kDirectA ? Traits::kStages + 1 : Traits::kStages) *
            Traits::kBlockM * Traits::kK +
        (kDirectB || kBStagePath ? Traits::kStages + 1 : Traits::kStages) *
            Traits::kBlockN * Traits::kK;
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
template <typename Traits, bool OutFp8 = false, 
        typename LayoutA = RowMajor, typename LayoutB = RowMajor, bool kGroupRaster = false,
        bool kBStaged = true>
__global__ void __launch_bounds__(Traits::kCtaThreads,
                                  Fp8GemmSmem<Traits, LayoutA, LayoutB,
                                              kBStaged>::kMinCtas) 
    fp8_gemm_kernel(FP8Params p) {
    using T8 = std::conditional_t<Traits::kIsE5M2, __nv_fp8_e5m2, __nv_fp8_e4m3>;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kK = Traits::kK;
    constexpr int kStages = Traits::kStages;
    constexpr int kCtaThreads = Traits::kCtaThreads;
    constexpr bool kCrossA = Fp8GemmSmem<Traits, LayoutA, LayoutB, kBStaged>::kCrossA;
    constexpr bool kCrossB = Fp8GemmSmem<Traits, LayoutA, LayoutB, kBStaged>::kCrossB;
    constexpr bool kBStagePath =
        Fp8GemmSmem<Traits, LayoutA, LayoutB, kBStaged>::kBStagePath;
    constexpr bool kDirectA = Fp8GemmSmem<Traits, LayoutA, LayoutB, kBStaged>::kDirectA;
    constexpr bool kDirectB = Fp8GemmSmem<Traits, LayoutA, LayoutB, kBStaged>::kDirectB;
    static_assert(kStages >= 1 && kStages <= 8,
                  "FP8 GEMM stages must be in [1, 8]");
    // Tiles are flat [rows * kK] with a 16B-chunk XOR swizzle (tile_at):
    // ldmatrix reads whole 16B chunks through the same mapping the staging
    // writes, and the swizzle removes the bank conflict the unswizzled
    // 8-word row stride caused (see tile_at). The stages live in dynamic
    // shared memory so deep pipelines (kStages * (kBlockM + kBlockN) * kK >
    // 48KB static limit) opt in via cudaFuncSetAttribute in the launcher.
    extern __shared__ __align__(16) char fp8_gemm_smem[];
    // Per operand: congruous = kStages rotating canonical buffers; direct-
    // crosswise = kStages+1 of them (the load for tile i+kStages targets
    // buffer (i-1)%(kStages+1) — the one compute(i-1) finished reading at
    // the previous barrier — so it issues right after barrier 1 and its
    // global-load latency overlaps the MMA phase below); staged-crosswise
    // (B) = kStages K-major staging buffers (filled by cp.async, one per
    // tile in flight) followed by one canonical buffer the per-tile
    // transpose rewrites.
    constexpr int kAStageBytes = kBlockM * kK;
    constexpr int kBStageBytes = kBlockN * kK;
    constexpr int kARing = kDirectA ? kStages + 1 : kStages;  // A canonic ring
    constexpr int kBRing = kDirectB ? kStages + 1 : kStages;  // B canonic ring
    constexpr int kStB = kStages;  // B staging ring size (see above)
    T8* const a_base = reinterpret_cast<T8*>(fp8_gemm_smem);
    T8* const b_base =
        reinterpret_cast<T8*>(fp8_gemm_smem + kARing * kAStageBytes);
    T8* const b_canon = b_base + kStB * kBStageBytes;  // staged B only

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
    // kGroupRaster is a template knob (the launcher defaults it to the
    // measured best per layout: grouped for A-crosswise (dW) and for the
    // congruous NT forward — whose big B operand gains the most from the
    // shared stripe — plain for dX's crosswise-B layouts, where it measured
    // neutral).
    constexpr int kGroupM = 8;
    int block_m, block_n;
    if constexpr (kGroupRaster) {
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
    const int64_t row_base = (int64_t)block_m * kBlockM + warp_m * 64 + group;
    const int64_t output_col =
        (int64_t)block_n * kBlockN + warp_n * 32 + thread_in_group * 2;
    const int a_row0 = warp_m * 64;  // + mt * 16 in the loop
    const int b_row0 = warp_n * 32;  // + nt * 8
    const float sa = *p.scale_a;
    const float sb = *p.scale_b;
    float acc[4][4][4] = {};  // [nt][mt][acc]

    // Both operands end up in the canonical [M][kK] / [N][kK] shared tiles
    // the MMA fragments read, regardless of their global layout. A's tag
    // already names the operand view ([M][K] = [rows][contract]); B's tag is
    // relative to the canonical [K][N], so the stage-load sees its transpose
    // (transpose_layout_t, see common.h). Congruous operands cp.async
    // straight into their rotating canonical buffers; crosswise operands
    // cp.async into K-major staging (zero transformation) and get a per-tile
    // smem->smem transpose below.
    // Asynchronous loads for tile `tile`: congruous operands cp.async into
    // their canonical rings, a staged B cp.asyncs into its K-major staging
    // ring. Called after the post-compute barrier, alongside the commit.
    auto load_async = [&](int64_t tile) {
        const int64_t k_base = tile * kK;
        if constexpr (!kDirectA)
            load_operand_tile<T8, kK, kBlockM, kCtaThreads>(
                a_base + (tile % kARing) * kAStageBytes, a, m, k, a_ld, tid,
                k_base, (int64_t)block_m * kBlockM);
        if constexpr (kBStagePath)
            stage_crosswise_tile<T8, kK, kBlockN, kCtaThreads>(
                b_base + (tile % kStB) * kBStageBytes, b, n, k, b_ld, tid,
                k_base, (int64_t)block_n * kBlockN);
        if constexpr (!kDirectB && !kBStagePath)
            load_operand_tile<T8, kK, kBlockN, kCtaThreads>(
                b_base + (tile % kBRing) * kBStageBytes, b, n, k, b_ld, tid,
                k_base, (int64_t)block_n * kBlockN);
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
    auto load_direct = [&](int64_t tile) {
        const int64_t k_base = tile * kK;
        if constexpr (kDirectA)
            load_crosswise_direct<T8, kK, kBlockM, kCtaThreads>(
                a_base + (tile % kARing) * kAStageBytes, a, m, k, a_ld, tid,
                k_base, (int64_t)block_m * kBlockM);
        if constexpr (kDirectB)
            load_crosswise_direct<T8, kK, kBlockN, kCtaThreads>(
                b_base + (tile % kBRing) * kBStageBytes, b, n, k, b_ld, tid,
                k_base, (int64_t)block_n * kBlockN);
    };
    // smem->smem transpose of one k_seg region (kSegQuads contract quads) of
    // this tile's staged-crosswise B into its single canonical buffer.
    auto transpose_tile = [&](int tile, int seg) {
        if constexpr (!kBStagePath) return;
        constexpr int kSegQuads = kK / 4 / (kK / kMmaK);  // quads per k_seg
        constexpr int kBRegion = kSegQuads * (kBlockN / 16);
        const T8* b_stg = b_base + (tile % kStB) * kBStageBytes;
        for (int idx = tid; idx < kBRegion; idx += kCtaThreads)
            transpose_crosswise_region<T8, kK, kBlockN>(b_canon, b_stg, idx,
                                                        seg * kSegQuads);
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
    // Direct loads run synchronously here (back to back with their commit);
    // the steady state below overlaps them with the compute phase.
#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
        if (stage < tile_count) {
            load_async(stage);
            load_direct(stage);
            astrai::cp_async_commit_group();
        }
    }

    for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const int64_t remaining = tile_count - tile_index - 1;

        // Keep up to kStages - 1 younger groups in flight while making the
        // oldest group (the current stage) ready for consumption.
        const int keep_groups =
            remaining < kStages - 1 ? static_cast<int>(remaining) : kStages - 1;
        astrai::cp_async_wait_group_dispatch<kStages - 1>(keep_groups);
        // Barrier 1: every thread's cp.async for this stage is complete
        // before any thread reads tiles written by other threads.
        __syncthreads();

        // Direct chunks for tile i+kStages: issue LDG+PRMT+STS now so the
        // global-load latency hides behind the MMA phase below.
        if (tile_index + kStages < tile_count)
            load_direct(tile_index + kStages);

        // Staged-crosswise B: produce the canonical tile one k_seg region at
        // a time so each region's transpose overlaps the previous region's
        // MMA sequence (the transposes are pure shared-memory traffic — B's
        // global path stayed fully asynchronous above).
        constexpr int kSegs = kK / kMmaK;
        if constexpr (kBStagePath) {
            transpose_tile(tile_index, 0);
            // Barrier 2: region 0 visible to every thread before its
            // fragment loads. (Compiled out for congruous/direct layouts.)
            __syncthreads();
        }

        const T8* a_tile = a_base + (size_t)(tile_index % kARing) * kAStageBytes;
        const T8* b_tile = kBStagePath
                               ? b_canon
                               : b_base + (size_t)(tile_index % kBRing) * kBStageBytes;

        // 4 ldmatrix.x2 (B) + 4 ldmatrix.x4 (A) feed 16 mma.sync per k_seg —
        // 0.5 load instructions per MMA, versus 4.5 scalar LDS per MMA in
        // the 128x64-tile version (the kernel was LSU-issue-bound there).
        // B fragments double-buffer across k_segs while B is congruous (no
        // region writes in flight); a crosswise B reloads per k_seg after
        // the region's transpose became visible.
        unsigned b_frag[2][4][2];
        if constexpr (!kBStagePath) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int row = b_row0 + nt * 8 + r7;
                astrai::ldmatrix_x2_lane(b_frag[0][nt],
                                         frag_addr<T8, kK>(b_tile, row, rh8));
            }
        }
#pragma unroll
        for (int k_seg = 0; k_seg < kSegs; ++k_seg) {
            const int bcur = k_seg & 1, bnext = bcur ^ 1;
            if constexpr (kBStagePath) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int row = b_row0 + nt * 8 + r7;
                    astrai::ldmatrix_x2_lane(
                        b_frag[bcur][nt],
                        frag_addr<T8, kK>(b_tile, row, k_seg * 2 + rh8));
                }
            } else if (k_seg + 1 < kSegs) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int row = b_row0 + nt * 8 + r7;
                    astrai::ldmatrix_x2_lane(
                        b_frag[bnext][nt],
                        frag_addr<T8, kK>(b_tile, row, (k_seg + 1) * 2 + rh8));
                }
            }
            // Region k_seg+1's transpose overlaps this region's MMA work
            // (disjoint canonical regions, no race).
            if constexpr (kBStagePath) {
                if (k_seg + 1 < kSegs)
                    transpose_tile(tile_index, k_seg + 1);
            }
            // Software-pipelined A fragments: the ldmatrix.x4 for row mt+1
            // is issued before the MMAs consuming row mt, so the LDS fixed
            // latency hides behind tensor-pipe work (cuts the `wait` stall,
            // ~2.3 cycles/issue before this). Costs 4 extra registers.
            unsigned a_frag[5][4];
            astrai::ldmatrix_x4_lane(
                a_frag[0], frag_addr<T8, kK>(a_tile, a_row0 + rh8 * 8 + r7,
                                             k_seg * 2 + rh16));
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                if (mt < 3)
                    astrai::ldmatrix_x4_lane(
                        a_frag[mt + 1],
                        frag_addr<T8, kK>(a_tile,
                                          a_row0 + (mt + 1) * 16 + rh8 * 8 + r7,
                                          k_seg * 2 + rh16));
#pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    astrai::mma_sync<T8>(acc[nt][mt], a_frag[mt], b_frag[bcur][nt],
                                         acc[nt][mt]);
            }
            // Barrier 3: region k_seg+1's transposes complete and become
            // visible before the next k_seg reads them.
            if constexpr (kBStagePath) {
                if (k_seg + 1 < kSegs) __syncthreads();
            }
        }
        // Barrier 4: every thread finished reading this stage's tiles before
        // the prefetch for the (i+kStages)-th tile overwrites them (and the
        // next iteration's transposes rewrite the canonical buffer).
        __syncthreads();
        if (tile_index + kStages < tile_count) {
            load_async(tile_index + kStages);
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
                if (col + 1 < n && (reinterpret_cast<uintptr_t>(dst) & 3) == 0) {
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
void launch_fp8_quantize(const FP8QuantizeParams& p, cudaStream_t stream) {
    constexpr int kThreads = 256;
    // One block per 256 vectors (8 elements each); at least one block so the
    // scalar tail of a tiny / misaligned tensor is still covered.
    int64_t blocks = (p.total / 8 + kThreads - 1) / kThreads;
    if (blocks < 1) blocks = 1;
    fp8_quantize_kernel<Fmt><<<blocks, kThreads, 0, stream>>>(p);
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
// per LayoutA (grouped for A-crosswise, plain for A-congruous). m <= 64
// dispatches to the 64x128 CTA — a 128-row CTA would waste half its MMA work
// on predicated-off rows.
// Crosswise B takes the asynchronous staging+transpose pipeline only when
// the contract dim is long enough that B streams from DRAM (dX-class GEMMs,
// k = N_ffn); short-K crosswise GEMMs (dW: k = M tokens) read L2-resident
// operands, where the staging round trip costs more shared-memory traffic
// than the latency it hides (measured: dW ~37 TF direct vs ~29 TF staged,
// dX ~39 TF staged vs ~38 direct).
constexpr int64_t kCrossStageMinK = 8192;

template <FP8Format Fmt, bool OutFp8 = false, typename LayoutA = RowMajor,
          typename LayoutB = RowMajor, int kK = 64, int Stages = 2,
          bool GroupRaster = std::is_same_v<LayoutA, ColMajor> || std::is_same_v<LayoutB, ColMajor>>
void launch_fp8_gemm(const FP8Params& p, cudaStream_t stream) {
    dim3 grid((p.n + 127) / 128, (p.m + 127) / 128);
    const bool b_staged = p.k >= kCrossStageMinK;
    if (p.m <= 64) {
        using Traits = Fp8GemmTraits<Fmt, 64, 128, kK, Stages>;
        if (b_staged)
            launch_with_smem<fp8_gemm_kernel<Traits, OutFp8, LayoutA, LayoutB,
                                             GroupRaster, true>>(
                Fp8GemmSmem<Traits, LayoutA, LayoutB, true>::kBytes, grid,
                dim3(Traits::kCtaThreads), stream, p);
        else
            launch_with_smem<fp8_gemm_kernel<Traits, OutFp8, LayoutA, LayoutB,
                                             GroupRaster, false>>(
                Fp8GemmSmem<Traits, LayoutA, LayoutB, false>::kBytes, grid,
                dim3(Traits::kCtaThreads), stream, p);
    } else {
        using Traits = Fp8GemmTraits<Fmt, 128, 128, kK, Stages>;
        if (b_staged)
            launch_with_smem<fp8_gemm_kernel<Traits, OutFp8, LayoutA, LayoutB,
                                             GroupRaster, true>>(
                Fp8GemmSmem<Traits, LayoutA, LayoutB, true>::kBytes, grid,
                dim3(Traits::kCtaThreads), stream, p);
        else
            launch_with_smem<fp8_gemm_kernel<Traits, OutFp8, LayoutA, LayoutB,
                                             GroupRaster, false>>(
                Fp8GemmSmem<Traits, LayoutA, LayoutB, false>::kBytes, grid,
                dim3(Traits::kCtaThreads), stream, p);
    }
}

}  // namespace fp8
}  // namespace astrai
