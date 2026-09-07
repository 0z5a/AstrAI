#pragma once
// FP8 GEMM umbrella: the kernel orchestrator and the host-side launch
// planning. Device layers live in gemm/ (policy / load / scheduler /
// mainloop / epilogue) — pure CUDA, no torch; launchers are plain functions
// shared by the torch binding and the C tests. Layout tags and the NN swap
// semantics are documented in common.h and the design notes
// (docs/developer/cuda_kernels.md).

#include <algorithm>
#include <atomic>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <type_traits>

#include "common/pipeline.cuh"
#include "common/device.cuh"
#include "common/launch.cuh"
#include "common/reduce.cuh"
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

// The ONE quantized-GEMM orchestrator (cp.async staging).
template <typename Policy>
__global__ void __launch_bounds__(Policy::kCtaThreads, Policy::kMinCtas)
    gemm_kernel(GemmParams p) {
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
                  Mainloop::RingA::Layout::kTotalBytes +
                  Mainloop::RingB::Layout::kTotalBytes,
                  "output tile must fit the reclaimed operand smem");
    const int2 bn = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    Mainloop mainloop(gemm_smem, a, b, p.m, p.n, p.k, p.a_ld, p.b_ld,
                      threadIdx.x, bn);
    typename Mainloop::AccTensor acc = {};  // C cells on the (mt, nt) grid
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


// TMA orchestrator (sm_90+, dual-congruous staging): identical rings,
// layouts and epilogue; the staging discipline changes — one elected
// thread arms a per-slot mbarrier and issues the operand boxes
// (cp.async.bulk.tensor), consumers wait the slot's phase. The rings sit
// on a 1024B-aligned base because TMA swizzles the ABSOLUTE shared
// address (the pad is budgeted in Policy::kSmemBytes), and the mbarriers
// live right past the B ring.
template <typename Policy, bool kRank3A, bool kRank3B>
__global__ void __launch_bounds__(Policy::kCtaThreads, Policy::kMinCtas)
    gemm_kernel_tma(GemmParams p, const __grid_constant__ CUtensorMap tma_a,
                    const __grid_constant__ CUtensorMap tma_b) {
    using Traits = typename Policy::Traits;
    using Mainloop = GemmCollectiveMainloop<Policy>;
    using Epilogue = GemmCollectiveEpilogue<Policy>;
    static_assert(!Mainloop::kDirectA && !Mainloop::kDirectB,
                  "TMA staging requires dual-congruous operands");
    extern __shared__ __align__(16) char gemm_smem[];
    // Round the ring base up to its 1024B pattern period. Two's-complement
    // form: already-aligned bases pad 0 (~p would pad 1023 and misalign).
    char* smem =
        gemm_smem + ((-reinterpret_cast<uintptr_t>(gemm_smem)) & 1023u);

    using OutT = typename Policy::OutT;
    auto* out = reinterpret_cast<OutT*>(p.out_ptr) +
                (int64_t)blockIdx.z * p.out_batch_stride;

    static_assert(Mainloop::kBlockM * Mainloop::kBlockN * sizeof(OutT) <=
                  Mainloop::RingA::Layout::kTotalBytes +
                  Mainloop::RingB::Layout::kTotalBytes,
                  "output tile must fit the reclaimed operand smem");

    GemmTmaContext<kRank3A, kRank3B> tma;
    tma.map_a = &tma_a;
    tma.map_b = &tma_b;
    tma.bars = reinterpret_cast<uint64_t*>(
        smem + Mainloop::RingA::Layout::kTotalBytes +
               Mainloop::RingB::Layout::kTotalBytes);
    tma.depth = Mainloop::kARing;
    tma.z = blockIdx.z;
    if (threadIdx.x == 0) {
        for (int s = 0; s < Mainloop::kARing; ++s) {
            astrai::mbarrier_init(tma.full(s), 1);  // producer expect_tx
            astrai::mbarrier_init(tma.empty(s), Policy::kCtaThreads);
        }
    }
    __syncthreads();

    const int2 bn = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    Mainloop mainloop(smem, static_cast<const typename Mainloop::ElemA*>(p.a_ptr),
                      static_cast<const typename Mainloop::ElemB*>(p.b_ptr), p.m,
                      p.n, p.k, p.a_ld, p.b_ld, threadIdx.x, bn);
    typename Mainloop::AccTensor acc = {};
    mainloop.prologue(tma);
    mainloop.accumulate(acc, tma);
    // No cp.async groups on this path; the CTA join alone releases the
    // rings for the epilogue's reclaim.
    __syncthreads();
    Epilogue(smem, p, bn.x, bn.y, threadIdx.x).run(acc, out);
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

// Experiment/debug knobs, one getenv at first use:
//   ASTR_GEMM_NO_TMA=1 forces the cp.async staging everywhere;
//   ASTR_GEMM_S3=1 flips the congruous scan to prefer the s3 deep rings
//   (stage-depth calibration — the cp.async rings measured it a wash on
//   RTX 5090, the TMA rings may price differently).
inline bool gemm_tma_disabled() {
    static const bool off = std::getenv("ASTR_GEMM_NO_TMA") != nullptr;
    return off;
}

inline bool gemm_prefer_s3() {
    static const bool on = std::getenv("ASTR_GEMM_S3") != nullptr;
    return on;
}

// ASTR_GEMM_NO_MX=1 keeps symmetric fp8 on the plain cell — the A/B knob
// for the sm_120 block_scale cell (MxMmaOp; env read once per process).
inline bool gemm_mx_disabled() {
    static const bool off = std::getenv("ASTR_GEMM_NO_MX") != nullptr;
    return off;
}

// Grid for one Policy's tile: N x M block count, batch on z.
template <typename Traits>
dim3 gemm_grid(const GemmParams& p) {
    return dim3((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
                (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
}

// One read-only plan-log line per launch (ASTR_GEMM_PLAN=1; " mx" marks the
// block_scale cell).
inline void log_gemm_plan(const GemmParams& p, const dim3& grid, int bm,
                          int bn, int stages, int smem, bool tma,
                          bool mx = false) {
    if (!gemm_plan_log()) return;
    std::fprintf(stderr,
                 "[gemm-plan] %lldx%lldx%lld b=%d -> tile %dx%d s%d%s%s "
                 "grid %dx%dx%d raster %d smem %d\n",
                 (long long)p.m, (long long)p.n, (long long)p.k, p.batch, bm,
                 bn, stages, tma ? " tma" : "", mx ? " mx" : "", grid.x,
                 grid.y, grid.z, p.raster, smem);
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
    // The CTA class is the tile manifest's dispatch key (policy.cuh).
    using Cta = TileClass;
    Cta cta;
    // Ring depth (kStages) this launch runs. The manifest default is 2;
    // the planner raises it to 3 where the dtype pair's thinner operands
    // leave smem headroom under the same 96KB budget (humming's
    // _fit_num_stages rule: deepest ring that fits).
    int stages;
    int raster;  // GemmParams::raster value this launch runs
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
// CTA, one row per dtype class — RTX 5090-measured (saturation medians at
// M >= 1024, via the direct-instantiation pattern of csrc/tests/fp8_test.cu
// — `launch_policy<GemmPolicy<..., TileXxx, ...>>`, no planner; the
// 1B/1B int8 pair adjusted down from
// its saturation medians (.95/1.0), which over-credit the finer tiles on
// large-N mid-M grids — .92 keeps the measured small-CTA wins while
// holding the big CTA on the largest-N band). They also absorb smem
// residency (co-resident CTAs share SM throughput), which is why the cost
// model carries no separate residency term. The scalars are the one
// device-dependent constant set:
// re-measure with that harness when porting.
enum class GemmPerfClass : int { kW16A16 = 0, kW8A16, kW8A8, kF8A8 };
constexpr double kPlanEff[4][2] = {  // [class]{narrow, small}
    {0.76, 0.58},  // W16A16: bf16 x bf16 — fat operands lose most to finer tiles
    {0.92, 0.77},  // W8A16: 2B x 1B mixed (incl. bf16 x fp8), bf16 mma via dequant
    {0.82, 0.92},  // W8A8: int8 x int8 — dequant mma runs issue-bound, small holds
    {0.82, 0.52},  // F8A8: fp8 via the sm_120 block_scale cell — finer tiles
                   // staging-bound at the doubled mma rate
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
// small, s2 ring first (RTX 5090-measured: the s3 deep rings ride even to
// -1.3% on cp.async staging — three buffers already hide the LDGSTS
// latency, unlike humming's TMA rings that want the depth; the s3
// siblings stay in the manifest for staging variants that price
// differently), and a challenger needs a >2% lead to displace the
// incumbent, so ties resolve to the bigger tile — the same bias the
// measured ladder encoded. The small recipe keeps its residency-aware
// stage rule: the 3-stage ring's heavier smem (a second resident CTA on
// the 2Bx2B pair) only pays once the small grid spans multiple waves.
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
        {GemmPlan::Cta::kBig128, 128, 128, 64, 3, -1},
        {GemmPlan::Cta::kNarrow128x64, 128, 64, 64, 2, 0},
        {GemmPlan::Cta::kNarrow128x64, 128, 64, 64, 3, 0},
        {GemmPlan::Cta::kSmall64, 64, 64, 64, 2, 1},
        {GemmPlan::Cta::kSmall64, 64, 64, 64, 3, 1},
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
        if (best == nullptr) {
            best = &r;
            best_cost = cost;
            continue;
        }
        // ASTR_GEMM_S3: the same CTA's s3 twin takes over on ties (the
        // stage-depth measurement knob); otherwise the >2% challenger
        // rule keeps the scan's first — the s2 ring — on ties.
        const bool deeper_twin = gemm_prefer_s3() && r.stages == 3 &&
                                 best->stages == 2 && best->cta == r.cta;
        if (cost < best_cost * 0.98 || deeper_twin) {
            best = &r;
            best_cost = cost;
        }
    }
    // The s2 small recipe (48KB) is the floor every supported device fits;
    // the guard keeps the planner a total function on any other geometry.
    if (best == nullptr) best = &kRecipes[4];
    const int64_t tiles_64 =
        p.batch * ((p.m + 63) / 64) * ((p.n + 63) / 64);
    const bool small = best->cta == GemmPlan::Cta::kSmall64;
    const bool s3_fits =
        ring_smem_bytes(64, 64, 64, 3, ba, bb) <= dev.smem_max;
    int stages = best->stages;
    if (small && !(s3_fits && tiles_64 > 2 * dev.sms)) stages = 2;
    return GemmPlan{best->cta, stages,
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
        return GemmPlan{GemmPlan::Cta::kSmall64, s3 && s3_fits ? 3 : 2,
                        plan_raster(p, 64, 64, ba, bb, dev)};
    };
    // Padding rules first: predication waste beats any wave-fill effect.
    if (small_cta_padding(p.m, p.n)) return small(crosswise_ops > 0);
    if (crosswise_ops > 0) {
        const int64_t tiles_128 =
            (int64_t)p.batch * ((p.m + 127) / 128) * ((p.n + 127) / 128);
        if (big_fits && tiles_128 >= (int64_t)dev.sms * 3 / 2) {
            return GemmPlan{GemmPlan::Cta::kBig128, 2,
                            plan_raster(p, 128, 128, ba, bb, dev)};
        }
        return small(true);
    }
    return plan_congruous(p, dev, ba, bb, perf);
}

// Grid + launch for one concrete Policy — the only place a GEMM kernel
// goes to the wire.
template <typename Policy>
void launch_policy(GemmParams p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    dim3 grid = gemm_grid<Traits>(p);
    log_gemm_plan(p, grid, Traits::kBlockM, Traits::kBlockN, Traits::kStages,
                  Policy::kSmemBytes, /*tma=*/false, Traits::kMxCell);
    launch_with_smem<gemm_kernel<Policy>>(
        Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p);
}

// ---------------------------------------------------------------------------
// TMA staging (sm_90+): descriptor build + the TMA twin of launch_policy.
// The descriptors are cached exact-match (tma.cuh), so steady-state calls
// with unchanged tensors and tile pay the encode once.
// ---------------------------------------------------------------------------

// Big-CTA output-reclaim feasibility: the epilogue scatters the output
// tile into the reclaimed operand rings, and a fat output (fp32, 4B/elem)
// cannot fit the 128x128 tile inside thin operand rings — one definition
// serves the cp.async and TMA dispatch twins alike.
template <typename ElemA, typename ElemB, typename OutT>
constexpr bool big_reclaim_fits() {
    return 128 * 128 * sizeof(OutT) <=
           ring_smem_bytes(128, 128, 64, 2, (int)sizeof(ElemA),
                           (int)sizeof(ElemB));
}

// Build both operand descriptors for one TMA Policy's geometry. Dim/stride
// units are bytes along the contract dim; the batch encodes as a third
// dimension only when it strides (a broadcast operand shares one 2D map's
// coordinates across grid.z). The swizzle mode, box extents and byte
// scaling all derive from the operand's declared staging layout (the
// TmaSwizzleOf / tma_spec trait layer in common/tma.cuh) — the same
// instances the fragment readers consume, so the map cannot drift from
// the staging.
template <typename Policy>
bool tma_maps_for(const GemmParams& p, const CUtensorMap** ma,
                  const CUtensorMap** mb) {
    using Mainloop = GemmCollectiveMainloop<Policy>;
    *ma = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemA, typename Mainloop::SmemLayoutA,
                        Mainloop::kBlockM>(p.a_ptr, p.m, p.k, p.a_ld, p.batch,
                                           p.a_batch_stride));
    *mb = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemB, typename Mainloop::SmemLayoutB,
                        Mainloop::kBlockN>(p.b_ptr, p.n, p.k, p.b_ld, p.batch,
                                           p.b_batch_stride));
    return *ma != nullptr && *mb != nullptr;
}

// TMA launch for one Policy; false (nothing launched) when an operand
// cannot be described — misaligned base/ld — so the caller falls back to
// the cp.async twin.
template <typename Policy>
bool launch_policy_tma(const GemmParams& p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    // The planner's feasibility gate prices rings only; the TMA budget
    // adds the alignment pad + barriers and can tip past the opt-in
    // ceiling on the fattest pair — fall back rather than fail.
    if (Policy::kSmemBytes > astrai::device_facts().smem_max) return false;
    const CUtensorMap *ma = nullptr, *mb = nullptr;
    if (!tma_maps_for<Policy>(p, &ma, &mb)) return false;
    dim3 grid = gemm_grid<Traits>(p);
    log_gemm_plan(p, grid, Traits::kBlockM, Traits::kBlockN, Traits::kStages,
                  Policy::kSmemBytes, /*tma=*/true, Traits::kMxCell);
    // Rank bits pick the kernel instantiation: a strided batch rides the
    // 3D emitters, a broadcast operand keeps its shared 2D map — the
    // per-stage 2D/3D issue pick compiles away either way.
    auto launch_rank = [&](auto rank3a, auto rank3b) {
        launch_with_smem<gemm_kernel_tma<Policy, decltype(rank3a)::value,
                                         decltype(rank3b)::value>>(
            Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p,
            *ma, *mb);
    };
    if (p.batch > 1 && p.a_batch_stride > 0 && p.b_batch_stride > 0)
        launch_rank(std::true_type{}, std::true_type{});
    else if (p.batch > 1 && p.a_batch_stride > 0)
        launch_rank(std::true_type{}, std::false_type{});
    else if (p.batch > 1 && p.b_batch_stride > 0)
        launch_rank(std::false_type{}, std::true_type{});
    else
        launch_rank(std::false_type{}, std::false_type{});
    return true;
}

// Manifest dispatch (CUTLASS builder-table style): the plan's (CTA class,
// depth bit) selects exactly one TileManifest entry — the || short-circuits
// — and the resolver maps its tile onto a concrete Policy and launches.
template <typename Manifest, typename Resolver>
bool dispatch_tile(const GemmPlan& plan, const Resolver& resolve) {
    return std::apply(
        [&plan, &resolve](auto... tiles) {
            return (... || (tile_class<decltype(tiles)>() == plan.cta &&
                            (decltype(tiles)::kStages >= 3) ==
                                (plan.stages >= 3) &&
                            resolve.template run<decltype(tiles)>()));
        },
        Manifest{});
}

// TMA ladder resolver. The gate in launch_plan already guarantees
// dual-congruous 1-/2-byte operands, so the fast tile stays; only the
// output-reclaim fallback swaps big -> narrow. std::conditional_t keeps
// every alias instantiable, which the kernel's reclaim static_assert
// requires (an if-constexpr branch still NAMES its dead types).
template <typename ElemA, typename ElemB, typename LayoutOut, typename OutT,
          bool kBigReclaim, bool UseMx = false>
struct TmaLauncher {
    const GemmParams& p;
    cudaStream_t stream;
    template <typename Tile>
    bool run() const {
        using TileT = std::conditional_t<
            tile_class<Tile>() != TileClass::kBig128, Tile,
            std::conditional_t<
                kBigReclaim, Tile,
                std::conditional_t<(Tile::kStages >= 3), TileNarrow128x64s3,
                                   TileNarrow128x64>>>;
        return launch_policy_tma<
            GemmPolicy<ElemA, ElemB, RowMajor, ColMajor, TileT, LayoutOut,
                       OutT, false, true, UseMx>>(p, stream);
    }
};

// cp.async ladder resolver. Two big-CTA substitutions: the fast loop exists
// only for dual-congruous staging (crosswise operands take the predicated
// generic loop — the NonFast twin), and a fat output (fp32, 4B) that cannot
// reclaim the big rings routes to the narrow CTA — same math at lower
// reuse. Narrow and small entries pass through.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut, typename OutT, bool kBigFast, bool kBigReclaim,
          bool UseMx = false>
struct CpAsyncLauncher {
    GemmParams p;
    cudaStream_t stream;
    template <typename Tile>
    bool run() const {
        using NonFast = GemmTileConfig<typename Tile::CtaShape,
                                       typename Tile::WarpShape, Tile::kStages,
                                       false>;
        using TileT = std::conditional_t<
            tile_class<Tile>() != TileClass::kBig128, Tile,
            std::conditional_t<
                kBigReclaim, std::conditional_t<kBigFast, Tile, NonFast>,
                std::conditional_t<(Tile::kStages >= 3), TileNarrow128x64s3,
                                   TileNarrow128x64>>>;
        launch_policy<GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, TileT,
                                 LayoutOut, OutT, false, false, UseMx>>(
            p, stream);
        return true;
    }
};

// Plan -> Policy: compose the operand facts with one manifest tile
// (dispatch_tile); CpAsyncLauncher applies this ladder's substitutions.
// plan.stages >= 3 selects the deep-ring sibling of the same geometry (the
// planner only raises it where the operand pair's ring fits the smem
// opt-in ceiling). Takes the params by value: the plan's raster decision
// lands in the copy the kernel receives (callers keep theirs). UseMx threads
// the block_scale cell through both ladders (launch_plan routes it).
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16,
          bool UseMx = false>
void launch_plan_impl(GemmParams p, const GemmPlan& plan,
                      cudaStream_t stream) {
    p.raster = plan.raster;
    constexpr bool kBigFast = !std::is_same_v<LayoutA, ColMajor> &&
                              !std::is_same_v<LayoutB, RowMajor>;
    // TMA staging first when the layout pair and dtypes allow it (the
    // planner's stage/tile decisions are shared): sm_90+ device, no
    // kill switch, and every descriptor encodable — else the cp.async
    // twin below runs unchanged.
    if constexpr (kBigFast && sizeof(ElemA) <= 2 && sizeof(ElemB) <= 2) {
        constexpr bool kTmaReclaim = big_reclaim_fits<ElemA, ElemB, OutT>();
        if (!gemm_tma_disabled() && astrai::device_facts().cc >= 90 &&
            dispatch_tile<TileManifest>(
                plan, TmaLauncher<ElemA, ElemB, LayoutOut, OutT, kTmaReclaim,
                                  UseMx>{p, stream}))
            return;
    }
    constexpr bool kBigReclaim = big_reclaim_fits<ElemA, ElemB, OutT>();
    dispatch_tile<TileManifest>(
        plan, CpAsyncLauncher<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                              kBigFast, kBigReclaim, UseMx>{p, stream});
}

// The planner entry: symmetric fp8 rides the sm_120 block_scale cell unless
// ASTR_GEMM_NO_MX knocks it out.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16>
void launch_plan(GemmParams p, const GemmPlan& plan, cudaStream_t stream) {
    constexpr bool kMxCell =
        (std::is_same_v<ElemA, __nv_fp8_e4m3> &&
         std::is_same_v<ElemB, __nv_fp8_e4m3>) ||
        (std::is_same_v<ElemA, __nv_fp8_e5m2> &&
         std::is_same_v<ElemB, __nv_fp8_e5m2>);
    if constexpr (kMxCell) {
        // cc is the CC-tens runtime form (120 = CC 12.0 — the convention
        // table lives at csrc/CMakeLists.txt's arch-level comment). This
        // gate is one half of a contract: the CMake side emits the
        // sm_120a SASS slice exactly when "120" is in the arch list, so
        // the route fires only where that image exists.
        if (!gemm_mx_disabled() && astrai::device_facts().cc == 120) {
            launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                             true>(p, plan, stream);
            return;
        }
    }
    launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT>(
        p, plan, stream);
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
        // NT (the fused-linear shape).
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
