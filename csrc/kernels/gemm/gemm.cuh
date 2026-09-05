#pragma once
// FP8 GEMM umbrella: the kernel orchestrator and the host-side launch
// planning. Device layers live in gemm/ (policy / load / scheduler /
// mainloop / epilogue) — pure CUDA, no torch; launchers are plain functions
// shared by the torch binding and the C tests. Layout tags and the NN swap
// semantics are documented in common.h and the design notes
// (docs/developer/cuda_kernels.md).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "common/cp_async.cuh"
#include "common/launch.cuh"
#include "epilogue.cuh"
#include "quantize/common.h"
#include "gemm/common.h"
#include "load.cuh"
#include "mainloop.cuh"
#include "policy.cuh"
#include "scheduler.cuh"

namespace astrai {
namespace gemm {

using quant::FP8Format;

template <typename Policy>
__global__ void __launch_bounds__(Policy::kCtaThreads, Policy::kMinCtas)
    gemm_kernel(GemmParams p) {
    using Traits = typename Policy::Traits;
    using Mainloop = GemmCollectiveMainloop<Policy>;
    using Epilogue = GemmCollectiveEpilogue<Policy>;
    // Stages live in dynamic shared memory so deep pipelines (> 48KB
    // static limit) opt in via cudaFuncSetAttribute in the launcher.
    extern __shared__ __align__(16) char gemm_smem[];

    // Batch slice (grid.z): broadcast operands carry a 0 stride, so the
    // same pointer serves every batch.
    using ElemA = typename Mainloop::ElemA;
    using ElemB = typename Mainloop::ElemB;
    using OutT = typename Policy::OutT;
    const ElemA* a = reinterpret_cast<const ElemA*>(p.a_ptr) +
                  (int64_t)blockIdx.z * p.a_batch_stride;
    const ElemB* b = reinterpret_cast<const ElemB*>(p.b_ptr) +
                  (int64_t)blockIdx.z * p.b_batch_stride;
    auto* out = reinterpret_cast<OutT*>(p.out_ptr) +
                (int64_t)blockIdx.z * p.out_batch_stride;

    static_assert(Mainloop::kBlockM * Mainloop::kBlockN * sizeof(OutT) <=
                  Mainloop::kARing * Mainloop::kBlockM * Mainloop::kK * sizeof(ElemA) +
                  Mainloop::kBRing * Mainloop::kBlockN * Mainloop::kK * sizeof(ElemB),
                  "output tile must fit the reclaimed operand smem");
    const int2 bn = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    Mainloop mainloop(gemm_smem, a, b, p.m, p.n, p.k, p.a_ld, p.b_ld,
                      threadIdx.x, bn);
    float acc[Mainloop::kNt][Mainloop::kMt][4] = {};  // [nt][mt][acc]
    mainloop.prologue();
    mainloop.accumulate(acc);
    // Drain the pipeline before the epilogue reclaims the operand rings.
    astrai::cp_async_wait_all();
    Epilogue(gemm_smem, p, bn.x, bn.y, threadIdx.x).run(acc, out);
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
    ASTRAI_LAUNCH_CHECK();
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
// without a GPU). Raster order is a plan field picked by the aspect
// heuristic (plan_raster); a manual p.raster=0 keeps plain raster
// reachable for experiments.
struct GemmPlan {
    enum class Cta { kSmall64, kNarrow128x64, kBig128 };
    Cta cta;
    bool small_s3;  // kSmall64 only: cp.async pipeline depth (2 vs 3 stages)
    int raster;     // GemmParams::raster value this launch runs
};

// Raster-order heuristic (CUTLASS's rule): walk the dimension with more
// tiles fastest. Groups chunk the long side, and the swept operand stays
// L2-resident across a group — consecutive CTAs share one stripe of the
// other operand. Width stays the measured 8.
inline int plan_raster(int64_t m, int64_t n, int bm, int bn) {
    return (m + bm - 1) / bm >= (n + bn - 1) / bn ? 8 : -8;
}

// crosswise_ops counts the operands taking the direct crosswise load
// (A ColMajor / B RowMajor storage): 0 = dual-congruous NT, 1 = TN and the
// NN swap, 2 = TT. The layout shifts the crossovers (measured tables in
// the design notes): the small CTA hides the crosswise LDG+PRMT latency
// far better, while the big CTA's operand reuse buys back load bandwidth
// the crosswise path does not traffic in.
inline GemmPlan plan_gemm(const GemmParams& p, int crosswise_ops = 0) {
    const int64_t sm = device_sm_count();
    const int64_t tiles_128 =
        (int64_t)p.batch * ((p.m + 127) / 128) * ((p.n + 127) / 128);
    const auto small = [&](bool s3) {
        return GemmPlan{GemmPlan::Cta::kSmall64, s3,
                        plan_raster(p.m, p.n, 64, 64)};
    };
    const auto big = [&]() {
        return GemmPlan{GemmPlan::Cta::kBig128, false,
                        plan_raster(p.m, p.n, 128, 128)};
    };
    const auto narrow = [&]() {
        return GemmPlan{GemmPlan::Cta::kNarrow128x64, false,
                        plan_raster(p.m, p.n, 128, 64)};
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
void launch_policy(const GemmParams& p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    dim3 grid((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
              (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
    launch_with_smem<gemm_kernel<Policy>>(
        Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p);
}

// Plan -> Policy: the production-tuned configs. Big CTA: 128x128 of 8 warps
// x 64x32, kK=64, 2-stage full ring, fast loop only for dual-congruous
// layouts. Narrow: 128x64. Small CTA: 64x64 of 4 warps x 32x32, kK=64,
// kFastLoop always on. ElemT and OutT are independent template knobs (both
// flow from the entry dispatch; fp8 formats arrive through fp8_elem_t).
// Takes the params by value: the plan's raster decision lands in the copy
// the kernel receives (callers keep theirs).
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16>
void launch_plan(GemmParams p, const GemmPlan& plan, cudaStream_t stream) {
    p.raster = plan.raster;
    constexpr bool kBigFast = !std::is_same_v<LayoutA, ColMajor> &&
                              !std::is_same_v<LayoutB, RowMajor>;
    // The epilogue reclaims the operand rings for the output tile; a fat
    // output (fp32, 4B/elem) cannot fit the 128x128 tile inside the fp8
    // rings (64KB > 48KB) — compile-time route those to the narrow CTA
    // (32KB tile <= 36KB rings), same math at lower reuse.
    constexpr int kRingBytes = 3 * 64 * (128 * (int)sizeof(ElemA) + 128 * (int)sizeof(ElemB));
    constexpr bool kBigReclaim = 128 * 128 * (int)sizeof(OutT) <= kRingBytes;
    switch (plan.cta) {
    case GemmPlan::Cta::kBig128: {
        if constexpr (kBigReclaim) {
            using Policy =
                GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                           128, 128, 64, 32, 64, 2, false, kBigFast>;
            launch_policy<Policy>(p, stream);
        } else {
            using Policy =
                GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                           128, 64, 32, 32, 64, 2, false, true>;
            launch_policy<Policy>(p, stream);
        }
        break;
    }
    case GemmPlan::Cta::kNarrow128x64: {
        using Policy =
            GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                       128, 64, 32, 32, 64, 2, false, true>;
        launch_policy<Policy>(p, stream);
        break;
    }
    case GemmPlan::Cta::kSmall64: {
        if (plan.small_s3) {
            using Policy =
                GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                           64, 64, 32, 32, 64, 3, false, true>;
            launch_policy<Policy>(p, stream);
        } else {
            using Policy =
                GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                           64, 64, 32, 32, 64, 2, false, true>;
            launch_policy<Policy>(p, stream);
        }
        break;
    }
    }
}

// Pure problem rewrite: the dual-N-contiguous problem (trans_a/trans_b both
// false) has no dedicated instantiation — it runs as its transpose
// E[N][M] = B^T @ A^T (CUTLASS-sm90's is_swapAB) over swapped operands,
// with the geometry-derived transposed epilogue staging scattering into
// [M][N] row-major buffer. The rewritten trans flags become the layout tags
// the launcher instantiates; the NN path pays a scalar-store scatter, which
// its rare usage makes the right trade.
inline void canonicalize_gemm(GemmParams& p, bool& trans_a, bool& trans_b) {
    if (!trans_a && !trans_b) {
        GemmParams s = p;  // E = B^T * A^T: swap roles, M <-> N
        s.m = p.n;
        s.n = p.m;
        s.a_ptr = p.b_ptr;
        s.b_ptr = p.a_ptr;
        s.a_ld = p.b_ld;
        s.b_ld = p.a_ld;
        s.a_batch_stride = p.b_batch_stride;
        s.b_batch_stride = p.a_batch_stride;
        // The caller's [M][N] buffer read as E = B^T A^T: the epilogue
        // walks the caller's rows (kernel n) with the caller's N stride.
        s.out_ld = p.n;
        p = s;
        trans_a = trans_b = true;
    }
}

// Dtype-generic entry point: canonicalize the problem, plan the launch,
// wire the layout tags through. ElemA / ElemB / OutT are independent
// knobs; fp8-format callers go through the wrapper below.
//
// Symmetric and mixed dtypes share this fan-out; the one asymmetry is NN
// (dual row-major storage): the swap rewrite exchanges operand roles and
// so assumes a single element type — symmetric operands rewrite to the
// transposed TT kernel, mixed operands instantiate the dual-row-major
// shape directly (A congruous, B crosswise) instead.
template <typename ElemA, typename ElemB = ElemA, typename OutT = __nv_bfloat16>
void gemm_dispatch(GemmParams p, cudaStream_t stream, bool trans_a,
                   bool trans_b) {
    constexpr bool kSymmetric = std::is_same_v<ElemA, ElemB>;
    bool swapped = false;
    if constexpr (kSymmetric) {
        swapped = !trans_a && !trans_b;  // canonicalize rewrites NN
        canonicalize_gemm(p, trans_a, trans_b);
    }
    // Crosswise operand count for the plan: transposed-A storage
    // (ColMajor) and plain-B storage (RowMajor) both take the direct
    // crosswise load.
    const int crosswise = (trans_a ? 1 : 0) + (trans_b ? 0 : 1);
    const GemmPlan plan = plan_gemm(p, crosswise);
    if (trans_a && trans_b) {
        // The swap computes the transposed problem; its (rewritten TT)
        // branch instantiates the column-major-output epilogue through
        // LayoutOut. Mixed never swaps, so its output stays row-major.
        if constexpr (kSymmetric) {
            if (swapped)
                launch_plan<ElemA, ElemB, ColMajor, ColMajor, ColMajor, OutT>(p, plan, stream);
            else
                launch_plan<ElemA, ElemB, ColMajor, ColMajor, RowMajor, OutT>(p, plan, stream);
        } else {
            launch_plan<ElemA, ElemB, ColMajor, ColMajor, RowMajor, OutT>(p, plan, stream);
        }
    } else if (trans_b) {
        launch_plan<ElemA, ElemB, RowMajor, ColMajor, RowMajor, OutT>(p, plan, stream);
    } else if (trans_a) {
        launch_plan<ElemA, ElemB, ColMajor, RowMajor, RowMajor, OutT>(p, plan, stream);
    } else {
        // Dual row-major: mixed only — symmetric NN was rewritten above
        // into the transposed TT kernel (if constexpr keeps this
        // instantiation out of symmetric builds).
        if constexpr (!kSymmetric)
            launch_plan<ElemA, ElemB, RowMajor, RowMajor, RowMajor, OutT>(p, plan, stream);
    }
}

// fp8-format entry over the generic dispatch (fp8_elem_t maps Fmt -> type;
// bf16 output is the fused-linear convention).
template <FP8Format Fmt>
void gemm(GemmParams p, cudaStream_t stream, bool trans_a, bool trans_b) {
    using ElemT = fp8_elem_t<Fmt>;
    gemm_dispatch<ElemT, ElemT>(p, stream, trans_a, trans_b);
}

// Non-template entry over the production instantiations (defined in
// launch.cu). Bindings that link it declare the extern-template
// suppressions locally before use, so C tests instantiating straight from
// this header stay single-file self-contained.
void launch_gemm(FP8Format fmt, const GemmParams& p, cudaStream_t stream,
                 bool trans_a, bool trans_b);

}  // namespace gemm
}  // namespace astrai
