#pragma once
// FP8 GEMM umbrella: the kernel orchestrator and the host-side launch
// planning. Device layers live in gemm/ (policy / load / scheduler /
// mainloop / epilogue) — pure CUDA, no torch; launchers are plain functions
// shared by the torch binding and the C tests. Layout tags and the NN swap
// semantics are documented in common.h and the design notes
// (docs/developer/cuda_kernels.md).

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "common/pipeline.cuh"
#include "common/device.cuh"
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
    // cp_async_wait_all drains only the CALLING thread's cp.asyncs, and
    // the final mainloop iteration carries no trailing barrier — without
    // this one, a thread racing into the epilogue scatters the output tile
    // over peers' still-in-flight staging writes (and their final fragment
    // reads). One barrier closes both windows.
    astrai::PipelineSync<Mainloop::kStages>{}.drain();
    Epilogue(gemm_smem, p, bn.x, bn.y, threadIdx.x).run(acc, out);
}

// ---------------------------------------------------------------------------
// Launchers — pure CUDA (no torch), usable from the binding and pure C tests.
// ---------------------------------------------------------------------------

// ASTR_GEMM_PLAN=1: read-only launch log (shape -> recipe / grid / raster)
// from launch_policy. One getenv at first use; nothing here can change the
// launch.
inline bool gemm_plan_log() {
    static const bool on = std::getenv("ASTR_GEMM_PLAN") != nullptr;
    return on;
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

// Raster order. Direction follows the tile aspect (walk the dimension with
// more tiles fastest, CUTLASS's rule): the N-side mirrored group keeps the
// measured width 8. The M-side group width is humming's L2-budget rule
// (tune/raster.py) instead of a fixed width: a group's A tiles are reused
// across its whole N sweep, so the group is sized to keep them L2-resident
// while B streams through the remainder — B already L2-resident means no
// grouping pays (g = 1, plain raster); otherwise reserve a B-streaming
// fraction of L2 (fatter B traffic than A reserves more), cap the group so
// the A side fits, and floor it at enough M rows to keep every SM busy
// within one group sweep.
inline int plan_raster(const GemmParams& p, int bm, int bn, int ba, int bb,
                       const DeviceFacts& dev) {
    const int64_t m_tiles = (p.m + bm - 1) / bm;
    const int64_t n_tiles = (p.n + bn - 1) / bn;
    if (m_tiles < n_tiles) return -8;
    if (p.n * p.k * (int64_t)bb <= dev.l2_bytes * 7 / 10) return 1;
    const double reserve = 0.12 + 0.28 * (double)bb / (double)ba;
    const double budget = (1.0 - std::min(reserve, 0.5)) * (double)dev.l2_bytes;
    const int64_t ub = (int64_t)(budget / ((double)bm * (double)p.k * ba));
    const int64_t lb = (dev.sms + n_tiles - 1) / n_tiles;
    int64_t g = std::min(ub, m_tiles);
    if (ub >= lb) g = std::min(std::max(g, lb), m_tiles);
    return (int)std::max(g, (int64_t)1);
}

// Per-SM throughput scalars of the non-big recipes relative to the big
// CTA, one row per dtype class — RTX 5090-measured (gemm_tile_bench.cu,
// saturation medians at M >= 1024; the 1B/1B int8 pair adjusted down from
// its saturation medians (.95/1.0), which over-credit the finer tiles on
// large-N mid-M grids — .92 keeps the measured small-CTA wins while
// holding the big CTA on the largest-N band). They also absorb smem
// residency (co-resident CTAs share SM throughput), which is why the cost
// model carries no separate residency term. The scalars are the one
// device-dependent constant set:
// re-measure with the harness when porting.
enum class GemmPerfClass : int { kW16A16 = 0, kW8A16, kW8A8, kF8A8 };
constexpr double kPlanEff[4][2] = {  // [class]{narrow, small}
    {0.76, 0.58},  // W16A16: bf16 x bf16 — fat operands lose most to finer tiles
    {0.92, 0.77},  // W8A16: 2B x 1B mixed (incl. bf16 x fp8), bf16 mma via dequant
    {0.82, 0.92},  // W8A8: int8 x int8 — dequant mma runs issue-bound, small holds
    {0.85, 0.56},  // F8A8: native fp8 k32 mma — small starves the issue slots
};

// Compile-time dtype-class derivation from the operand pair (the mma
// promotion rule plus operand widths; mixed bf16xfp8 lands with the 2B x
// 1B class — same bytes and same promoted bf16 k16 mma as W8A16).
template <typename ElemA, typename ElemB>
constexpr GemmPerfClass gemm_perf_class() {
    using MmaT = typename gemm_mma_traits<ElemA, ElemB>::MmaT;
    if constexpr (!std::is_same_v<MmaT, __nv_bfloat16>) {
        return GemmPerfClass::kF8A8;  // native fp8 symmetric pair
    } else if constexpr (std::is_same_v<ElemA, __nv_bfloat16> &&
                         std::is_same_v<ElemB, __nv_bfloat16>) {
        return GemmPerfClass::kW16A16;
    } else if constexpr (std::is_same_v<ElemA, int8_t> &&
                         std::is_same_v<ElemB, int8_t>) {
        return GemmPerfClass::kW8A8;
    } else {
        return GemmPerfClass::kW8A16;
    }
}

// Conguous (NT) path: a wave-count cost model over the manifest recipes
// replaces the measured crossover ladder — bands are derived from device
// arithmetic, so a new GPU needs no re-measured thresholds. A recipe's
// cost is ceil(tiles / sms) quantized waves of bm * bn / eff SM-work
// each, scaled by edge-tile padding waste. The scan runs big -> narrow ->
// small and a challenger needs a >2% lead to displace the incumbent, so
// ties resolve to the bigger tile — the same bias the measured ladder
// encoded. The small recipe's ring depth follows the wave count: the
// 3-stage ring's lighter smem (a second resident CTA) wins the sub-wave
// latency-bound band, the 4-stage ring's deeper pipeline wins once the
// small grid spans multiple waves.
inline GemmPlan plan_congruous(const GemmParams& p, const DeviceFacts& dev,
                               int ba, int bb, GemmPerfClass perf) {
    struct Recipe {
        GemmPlan::Cta cta;
        int bm, bn, k, stages;  // manifest geometry — a candidate's smem
                                // price derives from it
        int eff_idx;  // -1: the big CTA is the 1.0 reference
    };
    static constexpr Recipe kRecipes[] = {
        {GemmPlan::Cta::kBig128, 128, 128, 64, 2, -1},
        {GemmPlan::Cta::kNarrow128x64, 128, 64, 64, 2, 0},
        {GemmPlan::Cta::kSmall64, 64, 64, 64, 2, 1},
    };
    const double* eff = kPlanEff[(int)perf];
    const Recipe* best = nullptr;
    double best_cost = 0.0;
    for (const Recipe& r : kRecipes) {
        // Device-facts feasibility (humming's candidate filter): a recipe
        // over the smem opt-in ceiling cannot launch at all — prune before
        // scoring. Every manifest recipe fits on the production archs
        // (96KB max vs 99KB optin), so this only guards ports.
        if (ring_smem_bytes(r.bm, r.bn, r.k, r.stages, ba, bb) > dev.smem_max)
            continue;
        const int64_t tiles =
            p.batch * ((p.m + r.bm - 1) / r.bm) * ((p.n + r.bn - 1) / r.bn);
        const int64_t waves = (tiles + dev.sms - 1) / dev.sms;
        const double waste =
            (double)(((p.m + r.bm - 1) / r.bm) * r.bm *
                     ((p.n + r.bn - 1) / r.bn) * r.bn) /
            (double)(p.m * p.n);
        const double e = r.eff_idx < 0 ? 1.0 : eff[r.eff_idx];
        const double cost = (double)waves * r.bm * r.bn / e * waste;
        if (best == nullptr || cost < best_cost * 0.98) {
            best = &r;
            best_cost = cost;
        }
    }
    // The s2 small recipe (48KB) is the floor every supported device fits;
    // the guard keeps the planner a total function on any other geometry.
    if (best == nullptr) best = &kRecipes[2];
    const int64_t tiles_64 =
        p.batch * ((p.m + 63) / 64) * ((p.n + 63) / 64);
    const bool small = best->cta == GemmPlan::Cta::kSmall64;
    const bool s3_fits =
        ring_smem_bytes(64, 64, 64, 3, ba, bb) <= dev.smem_max;
    return GemmPlan{best->cta,
                    small && s3_fits && tiles_64 > 2 * dev.sms,
                    plan_raster(p, best->bm, best->bn, ba, bb, dev)};
}

// crosswise_ops counts the operands taking the direct crosswise load
// (A ColMajor / B RowMajor storage): 0 = dual-congruous NT, 1 = TN and the
// NN swap, 2 = TT. ba / bb are the operand element sizes; perf is the
// dtype class picking the planner's eff row. The padding gate and the
// crosswise ladder stay measured rules (the crosswise load path prices
// differently: the small CTA hides its LDG+PRMT latency, the big CTA's
// operand reuse wins once its grid fills ~1.5 waves); the congruous path
// runs the wave-count model.
inline GemmPlan plan_gemm(const GemmParams& p, int ba, int bb,
                          GemmPerfClass perf, int crosswise_ops = 0) {
    const DeviceFacts dev = device_facts();
    // Feasibility gates shared by the branches below: a recipe over the
    // device's smem opt-in ceiling demotes to the next fitting geometry
    // instead of failing the launch. Both are always true on the
    // production archs.
    const bool big_fits =
        ring_smem_bytes(128, 128, 64, 2, ba, bb) <= dev.smem_max;
    const bool s3_fits =
        ring_smem_bytes(64, 64, 64, 3, ba, bb) <= dev.smem_max;
    const auto small = [&](bool s3) {
        return GemmPlan{GemmPlan::Cta::kSmall64, s3 && s3_fits,
                        plan_raster(p, 64, 64, ba, bb, dev)};
    };
    // Padding rules first: predication waste beats any wave-fill effect.
    if (small_cta_padding(p.m, p.n)) return small(crosswise_ops > 0);
    if (crosswise_ops > 0) {
        const int64_t tiles_128 =
            (int64_t)p.batch * ((p.m + 127) / 128) * ((p.n + 127) / 128);
        if (big_fits && tiles_128 >= (int64_t)dev.sms * 3 / 2) {
            return GemmPlan{GemmPlan::Cta::kBig128, false,
                            plan_raster(p, 128, 128, ba, bb, dev)};
        }
        return small(true);
    }
    return plan_congruous(p, dev, ba, bb, perf);
}

// Grid + launch for one concrete Policy — the only place a GEMM kernel goes
// to the wire.
template <typename Policy>
void launch_policy(const GemmParams& p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    dim3 grid((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
              (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
    if (gemm_plan_log()) {
        std::fprintf(stderr,
                     "[gemm-plan] %lldx%lldx%lld b=%d -> tile %dx%d s%d "
                     "grid %dx%dx%d raster %d smem %d\n",
                     (long long)p.m, (long long)p.n, (long long)p.k, p.batch,
                     Traits::kBlockM, Traits::kBlockN, Traits::kStages,
                     grid.x, grid.y, grid.z, p.raster, Policy::kSmemBytes);
    }
    launch_with_smem<gemm_kernel<Policy>>(
        Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p);
}

// Plan -> Policy: compose the operand facts with one named tile config
// from the manifest in policy.cuh. The big CTA's fast loop exists only for
// dual-congruous staging (both operands cp.async); crosswise operands take
// the predicated generic loop. Takes the params by value: the plan's raster
// decision lands in the copy the kernel receives (callers keep theirs).
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16>
void launch_plan(GemmParams p, const GemmPlan& plan, cudaStream_t stream) {
    p.raster = plan.raster;
    constexpr bool kBigFast = !std::is_same_v<LayoutA, ColMajor> &&
                              !std::is_same_v<LayoutB, RowMajor>;
    using BigTile = std::conditional_t<kBigFast, TileBigFast, TileBig128x128>;
    // The epilogue reclaims the operand rings for the output tile; a fat
    // output (fp32, 4B/elem) cannot fit the 128x128 tile inside the fp8
    // rings (64KB > 48KB) — compile-time route those to the narrow CTA
    // (32KB tile <= 36KB rings), same math at lower reuse.
    constexpr int kRingBytes = ring_smem_bytes(
        128, 128, 64, 2, (int)sizeof(ElemA), (int)sizeof(ElemB));
    constexpr bool kBigReclaim = 128 * 128 * (int)sizeof(OutT) <= kRingBytes;
    using BigOrNarrow =
        std::conditional_t<kBigReclaim, BigTile, TileNarrow128x64>;
    switch (plan.cta) {
    case GemmPlan::Cta::kBig128:
        launch_policy<GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, BigOrNarrow,
                                 LayoutOut, OutT>>(p, stream);
        break;
    case GemmPlan::Cta::kNarrow128x64:
        launch_policy<GemmPolicy<ElemA, ElemB, LayoutA, LayoutB,
                                 TileNarrow128x64, LayoutOut, OutT>>(p, stream);
        break;
    case GemmPlan::Cta::kSmall64:
        if (plan.small_s3) {
            launch_policy<GemmPolicy<ElemA, ElemB, LayoutA, LayoutB,
                                     TileSmall64s3, LayoutOut, OutT>>(p, stream);
        } else {
            launch_policy<GemmPolicy<ElemA, ElemB, LayoutA, LayoutB,
                                     TileSmall64s2, LayoutOut, OutT>>(p, stream);
        }
        break;
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
    const GemmPlan plan = plan_gemm(p, (int)sizeof(ElemA), (int)sizeof(ElemB),
                                    gemm_perf_class<ElemA, ElemB>(),
                                    crosswise);
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

}  // namespace gemm
}  // namespace astrai
