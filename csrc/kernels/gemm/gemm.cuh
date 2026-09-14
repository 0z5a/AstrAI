#pragma once
// GEMM-family umbrella (bf16 / int8 / fp8): the kernel orchestrator and the
// host-side launch planning. Device layers live in gemm/ (policy / load /
// scheduler / mainloop / epilogue) — pure CUDA, no torch; launchers are
// plain functions shared by the torch binding and the C tests. Layout tags
// and the NN swap semantics live in common.h and
// docs/developer/cuda_kernels.md.

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <type_traits>

#include "common/pipeline.cuh"
#include "common/device.cuh"
#include "common/launch.cuh"
#include "epilogue.cuh"
#include "gemm/common.h"
#include "gemm/plan_table.h"
#include "mainloop.cuh"
#include "policy.cuh"
#include "scheduler.cuh"

namespace astrai {
namespace gemm {

// The ONE quantized-GEMM orchestrator (cp.async staging).
template <typename Policy>
__global__ void __launch_bounds__(Policy::kCtaThreads, Policy::kMinCtas)
    gemm_kernel(GemmParams p) {
    using Mainloop = GemmCollectiveMainloop<Policy>;
    using Epilogue = GemmCollectiveEpilogue<Policy>;
    // Stages live in dynamic shared memory so deep pipelines (> 48KB static
    // limit) opt in via cudaFuncSetAttribute in the launcher.
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

    static_assert(Mainloop::kOutputReclaimsRings,
                  "output tile must fit the reclaimed operand smem");
    const int2 blk = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    Mainloop mainloop(gemm_smem, a, b, p.m, p.n, p.k, p.a_ld, p.b_ld,
                      threadIdx.x, blk);
    typename Mainloop::AccTensor acc = {};  // C cells on the (mt, nt) grid
    mainloop.prologue();
    mainloop.accumulate(acc);
    // Drain the pipeline before the epilogue reclaims the operand rings:
    // cp_async_wait_all drains only the CALLING thread's copies and the
    // last mainloop iteration carries no trailing barrier — without this,
    // a thread racing into the epilogue scatters the output tile over
    // peers' still-in-flight staging writes. One barrier closes both.
    astrai::PipelineSync<Mainloop::kStages>{}.drain();
    Epilogue(gemm_smem, p, blk.x, blk.y, threadIdx.x).run(acc, out);
}


// TMA orchestrator (sm_90+, dual-congruous staging): identical rings,
// layouts and epilogue; the staging discipline changes — one elected
// thread arms a per-slot mbarrier and issues the operand boxes
// (cp.async.bulk.tensor), consumers wait the slot's phase. The rings sit
// on a 1024B-aligned base because TMA swizzles the ABSOLUTE smem address
// (the pad is budgeted in Policy::kSmemBytes), and the mbarriers live
// right past the B ring.
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

    static_assert(Mainloop::kOutputReclaimsRings,
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

    const int2 blk = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    Mainloop mainloop(smem, static_cast<const typename Mainloop::ElemA*>(p.a_ptr),
                      static_cast<const typename Mainloop::ElemB*>(p.b_ptr), p.m,
                      p.n, p.k, p.a_ld, p.b_ld, threadIdx.x, blk);
    typename Mainloop::AccTensor acc = {};
    mainloop.prologue(tma);
    mainloop.accumulate(acc, tma);
    // No cp.async groups on this path; the CTA join alone releases the
    // rings for the epilogue's reclaim.
    __syncthreads();
    Epilogue(smem, p, blk.x, blk.y, threadIdx.x).run(acc, out);
}

// ---------------------------------------------------------------------------
// Launchers — pure CUDA (no torch), usable from the binding and pure C tests.
// The runtime knobs (plan log, planner rank, staging A/B switches, table
// mode) live in plan_table.h's GemmConfig, seeded once from the deprecated
// ASTR_GEMM_* variables and owned at runtime by astrai.extension.plan.
// ---------------------------------------------------------------------------

// Grid for one Policy's tile: N x M block count, batch on z.
template <typename Traits>
dim3 gemm_grid(const GemmParams& p) {
    return dim3((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
                (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
}

// One read-only plan-log line per launch (plan.set_log; " mx" marks the
// block_scale cell).
inline void log_gemm_plan(const GemmParams& p, const dim3& grid, int bm,
                          int bn, int stages, int smem, bool tma,
                          bool mx = false) {
    if (!gemm_plan_log_enabled()) return;
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
inline int plan_raster(const PlanQuery& q, int bm, int bn) {
    const int64_t m_tiles = (q.m + bm - 1) / bm;
    const int64_t n_tiles = (q.n + bn - 1) / bn;
    if (m_tiles < n_tiles) return -8;
    if (q.n * q.k * (int64_t)q.bb <= q.dev.l2_bytes * 7 / 10) return 1;
    const double reserve = 0.12 + 0.28 * (double)q.bb / (double)q.ba;
    const double budget = (1.0 - std::min(reserve, 0.5)) * (double)q.dev.l2_bytes;
    const int64_t ub = (int64_t)(budget / ((double)bm * (double)q.k * q.ba));
    const int64_t lb = (q.dev.sms + n_tiles - 1) / n_tiles;
    int64_t g = std::min(ub, m_tiles);
    if (ub >= lb) g = std::min(std::max(g, lb), m_tiles);
    return (int)std::max(g, (int64_t)1);
}

// Dtype-class ids the plan-table rows key on.
enum class GemmPerfClass : int { kW16A16 = 0, kW8A16, kW8A8, kF8A8 };


// Compile-time dtype-class derivation from the operand pair (the mma
// promotion rule plus operand widths; mixed bf16xfp8 lands with the 2B x
// 1B class — same bytes and same promoted bf16 k16 mma as W8A16).
//
// The int8 pair is tested first: it promotes to an int8 mma, not to bf16, so
// the "not bf16 -> fp8 pair" arm below would otherwise swallow it and every
// W8A8 row would be unreachable (int8 dispatched on the F8A8 rows, and the
// plan log showed an int8 problem resolving to a class-3 row).
template <typename ElemA, typename ElemB>
constexpr GemmPerfClass gemm_perf_class() {
    using MmaT = typename gemm_mma_traits<ElemA, ElemB>::MmaT;
    if constexpr (std::is_same_v<ElemA, int8_t> &&
                  std::is_same_v<ElemB, int8_t>) {
        return GemmPerfClass::kW8A8;
    } else if constexpr (!std::is_same_v<MmaT, __nv_bfloat16>) {
        return GemmPerfClass::kF8A8;  // native fp8 symmetric pair
    } else if constexpr (std::is_same_v<ElemA, __nv_bfloat16> &&
                         std::is_same_v<ElemB, __nv_bfloat16>) {
        return GemmPerfClass::kW16A16;
    } else {
        return GemmPerfClass::kW8A16;
    }
}

static_assert(gemm_perf_class<int8_t, int8_t>() == GemmPerfClass::kW8A8,
              "int8 x int8 is its own class (see the ordering note above)");
static_assert(gemm_perf_class<__nv_fp8_e4m3, __nv_fp8_e4m3>() ==
                  GemmPerfClass::kF8A8,
              "fp8 x fp8 keys the F8A8 table");
static_assert(gemm_perf_class<__nv_bfloat16, __nv_bfloat16>() ==
                  GemmPerfClass::kW16A16,
              "bf16 x bf16 keys the W16A16 table");
static_assert(gemm_perf_class<__nv_bfloat16, int8_t>() ==
                  GemmPerfClass::kW8A16,
              "a quantized weight against bf16 activations keys W8A16");



// ---------------------------------------------------------------------------
// Recipe vocabulary: ONE spelling of a launchable tile configuration. The
// manifest types are the compiled truth, GemmRecipe is their runtime form,
// and every consumer — row tables, the analytical model, the launchers,
// the tile_vocabulary binding — names tiles through it.
// ---------------------------------------------------------------------------

struct GemmRecipe {
    int cta;      // TileClass ordinal — the row-file serialization key
    int stages;   // ring depth
    int kk;       // k-tile depth (the kK twins are separate recipes)
    int bm, bn;   // CTA geometry
    int threads;  // the manifest entry's warp tiling (first match wins)
    int smem;     // ring bytes at this staging pair's operand widths
};

// Deduped on the dispatch key: two tiles sharing (class, stages, kK) — the
// 16-warp small CTA behind its 32-warp twin — are one candidate, because
// dispatch_tile takes the first manifest match. smem prices against the
// operand widths the caller asks about, so a vocabulary is pair-specific.
template <typename Tile>
inline void append_recipe(std::vector<GemmRecipe>& out, int ba, int bb) {
    const GemmRecipe r{
        (int)tile_class<Tile>(), Tile::kStages, (int)Tile::CtaShape::kK,
        Tile::CtaShape::kM, Tile::CtaShape::kN,
        (Tile::CtaShape::kM / Tile::WarpShape::kM) *
            (Tile::CtaShape::kN / Tile::WarpShape::kN) * 32,
        ring_smem_bytes(Tile::CtaShape::kM, Tile::CtaShape::kN,
                        Tile::CtaShape::kK, Tile::kStages, ba, bb)};
    for (const GemmRecipe& have : out)
        if (have.cta == r.cta && have.stages == r.stages && have.kk == r.kk)
            return;
    out.push_back(r);
}

template <typename Manifest>
inline void collect_recipes(std::vector<GemmRecipe>& out, int ba, int bb) {
    std::apply(
        [&out, ba, bb](auto... tiles) {
            (append_recipe<decltype(tiles)>(out, ba, bb), ...);
        },
        Manifest{});
}

// Every recipe the ladders instantiate for one staging pair — the runtime
// half of manifest_for's rule (manifest_kind over crosswise + widths).
inline std::vector<GemmRecipe> gemm_recipes_for(bool crosswise_staging,
                                                int ba, int bb) {
    std::vector<GemmRecipe> out;
    switch (manifest_kind(crosswise_staging, ba, bb)) {
        case ManifestKind::kTwoByte:
            collect_recipes<TileManifest>(out, ba, bb);
            break;
        case ManifestKind::kByte:
            collect_recipes<TileManifestByte>(out, ba, bb);
            break;
        default:  // kCrosswise is the fallback kind, manifest_for included
            collect_recipes<TileManifestCross>(out, ba, bb);
            break;
    }
    return out;
}

// The instantiation oracle: does this staging pair's ladder carry a tile
// for (class, stages, kK)? This is the ONE gate a row's recipe fields must
// pass — the wide CTA's 1-byte-only rule, kK=32's dual-2-byte line
// requirement, ring depths past s3, the crosswise ladder's conservative
// set — because a row naming a non-instantiable combination would match no
// tile in dispatch_tile and launch nothing at all.
inline std::optional<GemmRecipe> recipe_of(int cta, int stages, int kk,
                                           bool crosswise, int ba, int bb) {
    for (const GemmRecipe& r : gemm_recipes_for(crosswise, ba, bb))
        if (r.cta == cta && r.stages == stages && r.kk == kk) return r;
    return std::nullopt;
}

// One dispatch decision: the recipe, the resolved raster (a row's literal,
// or plan_raster at the recipe's geometry), and the planner that made it.
// source is that planner's own name — the log line and the probe dict
// report it verbatim.
struct PlanDecision {
    GemmRecipe recipe;
    int raster;
    const char* source;
};

// The [gemm-plan] decision line (plan.set_log gates it; gen_plan_table's
// tag regex reads it).
inline void log_dispatch(const PlanQuery& q, const PlanDecision& d) {
    if (!gemm_plan_log_enabled()) return;
    std::fprintf(stderr,
                 "[gemm-plan] %s m%lld n%lld k%lld b=%d -> cta%d s%d "
                 "raster %d\n",
                 d.source, (long long)q.m, (long long)q.n, (long long)q.k,
                 (int)q.batch, d.recipe.cta, d.recipe.stages, d.raster);
}

// ---------------------------------------------------------------------------
// Planners: one plan strategy per dispatch source, composed into a chain
// (first planner to answer wins). A new source is one class and one chain
// entry; the mode knob (GemmConfig::planner) only picks which chain.
// ---------------------------------------------------------------------------
struct GemmPlanner {
    virtual ~GemmPlanner() = default;
    virtual const char* name() const = 0;
    virtual std::optional<PlanDecision> plan(const PlanQuery& q) const = 0;
};

// Rows to a decision: match (band + wave gates, all inside plan_row_for),
// then the instantiation oracle, then the smem ceiling — a stale tuning
// row falls through to the next planner instead of failing a launch.
// Raster 0 on the row resolves through plan_raster at the recipe's
// geometry (a bare 0 would be GemmParams' PLAIN raster, which costs ~14%
// on the M<<N shapes).
class RowSetPlanner final : public GemmPlanner {
public:
    using RowFn =
        std::function<std::optional<TableRow>(const PlanQuery&)>;
    RowSetPlanner(const char* source, RowFn rows, bool respects_table_off)
        : source_(source), rows_(std::move(rows)),
          respects_table_off_(respects_table_off) {}
    const char* name() const override { return source_; }
    std::optional<PlanDecision> plan(const PlanQuery& q) const override {
        if (respects_table_off_ && gemm_table_off()) return std::nullopt;
        const std::optional<TableRow> row = rows_(q);
        if (!row) return std::nullopt;
        const std::optional<GemmRecipe> recipe =
            recipe_of((int)row->cta, row->stages, row->kk, q.crosswise > 0,
                      q.ba, q.bb);
        if (!recipe || recipe->smem > q.dev.smem_max) return std::nullopt;
        return PlanDecision{
            *recipe,
            row->raster != 0 ? row->raster
                             : plan_raster(q, recipe->bm, recipe->bn),
            source_};
    }

private:
    const char* source_;
    RowFn rows_;
    bool respects_table_off_;
};

// The analytical planner — a port of DeepGEMM's SM90 heuristic
// (csrc/jit_kernels/heuristics/sm90.hpp, get_layout_info): candidates are
// priced by the traffic they move through L1 and L2, the only rates that
// VARY across tile choices (total FLOPs and HBM bytes are tile-invariant
// and cancel in the comparison). The wave term is the tail model:
// blocks spread over waves * sms * resident slots, so a grid that fills
// 2.02 waves pays half its time at ~2% fill — the measured failure mode
// of a literal-M table row on a part the literal was not calibrated for
// (M=512 x N=11008 picking a 128x128 CTA = 3 waves at 0.675 fill where
// the 128x64 twin runs 5 waves at 0.83 and measures +18%). No cluster or
// TMA multicast exists here, so the L2 term keeps the plain (bm + bn)
// reuse shape and no bytes are divided by a cluster factor.
//
// The per-cycle rates are DeepGEMM's H100-class constants; within one
// architecture they only weight the max(l1, l2) balance, and the winner
// is decided by ratios that survive a constant-factor error.
class ModelPlanner final : public GemmPlanner {
public:
    const char* name() const override { return "model"; }

    std::optional<PlanDecision> plan(const PlanQuery& q) const override {
        if (q.dev.sms <= 0 || q.m <= 0 || q.n <= 0 || q.k <= 0)
            return std::nullopt;
        const std::vector<GemmRecipe> recipes =
            gemm_recipes_for(q.crosswise > 0, q.ba, q.bb);
        double best = -1.0;
        std::optional<PlanDecision> best_d;
        for (const GemmRecipe& r : recipes) {
            const double cycles = price(r, q);
            if (cycles < 0.0) continue;
            // Ties go to the bigger CTA class then the deeper ring K (the
            // sweep generator's stable big > narrow > small preference plus
            // kK=64's loop amortization), so equal-cost scans do not
            // flip-flop with the manifest order. kK is otherwise invisible
            // to the model — the twins move identical bytes and (on parts
            // where both are 1-resident) identical slots — which is the
            // measured-not-modeled axis the row tables own.
            const bool tie_better =
                !best_d || r.cta > best_d->recipe.cta ||
                (r.cta == best_d->recipe.cta && r.kk > best_d->recipe.kk);
            if (best < 0.0 || cycles < best * (1.0 - 1e-9) ||
                (cycles <= best * (1.0 + 1e-9) && tie_better)) {
                best = cycles;
                best_d = PlanDecision{r, plan_raster(q, r.bm, r.bn),
                                      name()};
            }
        }
        return best_d;
    }

private:
    // Per-arch dense-bf16 tensor-pipe peak, FLOP/cycle/SM — the hardware
    // constant of the tc term, in the same spirit as DeepGEMM's per-cycle
    // L1/L2 rates. sm_120 measured ~508 (206 TF at M=2048 up_gate);
    // unlisted parts take the conservative default (the term only shifts
    // the max(l1, l2, tc) balance, and a slight miss scales every
    // candidate alike). The 1-byte classes double it (measured issue 506
    // vs 1011 TF on the s8 / block-scale cells); W8A16 rides the bf16
    // cell (the dequant insert is not the bottleneck — the doc's own
    // measurement).
    static double tc_peak(int cc, int perf_class) {
        (void)cc;
        return perf_class >= 2 ? 1024.0 : 512.0;
    }

    // Per-class tensor-pipe efficiency against that peak — the surviving
    // descendant of the retired kPlanEff table, reduced to the one axis
    // the model cannot derive: the mma issue density of a warp's own
    // tile. The 64x64 CTA's W16x32 warp tile issues 4 mma per k-step (a
    // 16-row A fragment amortized over one m16n8k16 row block) where the
    // narrow twin issues 8 and the big 16, and below ~8 mma/step the pipe
    // cannot be kept fed — measured as a flat 2x on the huge shapes
    // (9.02 vs 4.37 ms at 2048x28672x8192, everything else equal), which
    // no bytes/wave term prices. 0.5 for the small class, unity for the
    // rest; DeepGEMM never faces this because its JIT absorbs
    // per-config efficiency into the compiled kernel rather than the
    // planner.
    static double tc_eff(int cta) {
        return cta == (int)TileClass::kSmall64 ? 0.5 : 1.0;
    }

    static double price(const GemmRecipe& r, const PlanQuery& q) {
        const DeviceFacts& dev = q.dev;
        const int resident = dev.smem_per_sm > 0 && dev.regs_per_sm > 0
                                 ? std::min(dev.smem_per_sm / r.smem,
                                            min_ctas_for_ring(r.smem))
                                 : 0;
        if (resident <= 0) return -1.0;  // ring cannot be resident
        const int64_t blocks =
            ((q.m + r.bm - 1) / r.bm) * ((q.n + r.bn - 1) / r.bn) *
            (int64_t)q.batch;
        const int64_t slots = (int64_t)dev.sms * resident;
        const int64_t waves = (blocks + slots - 1) / slots;
        const double wave_eff =
            (double)blocks / (double)(waves * slots);

        // Per-block bytes: operand staging (L1 and L2), the fragment
        // reads the mma pipe makes from smem (L1; the 64-row floor is
        // DeepGEMM's wgmma_m term), and the output tile the epilogue
        // pushes back through L1/L2.
        const int64_t ab =
            q.k * ((int64_t)r.bm * q.ba + (int64_t)r.bn * q.bb);
        const int64_t cd =
            (int64_t)r.bm * r.bn * 2;  // bf16-out (fp32-out underprices cd)
        const int64_t tc =
            q.k * ((int64_t)(r.bm < 64 ? 64 : r.bm) * q.ba +
                   (int64_t)r.bn * q.bb) + cd;
        const double l2_bw =
            std::min(64.0 * dev.sms, 8e6 / 1.3e3);  // B/cycle
        const double l1_bw = 128.0 * dev.sms;        // B/cycle
        const double l2_cycles = (double)(ab + cd) * (double)blocks / l2_bw;
        const double l1_cycles =
            (double)(ab + tc + cd) * (double)blocks / l1_bw;
        // The tensor-pipe term DeepGEMM leaves implicit (their kernels
        // sit at L1/L2 limits, so FLOPs "cancel"). Here the mma.sync pipe
        // is the binding resource at the fat shapes — without this term
        // the model prices only reuse and over-picks the biggest CTA
        // (measured: 1b shapes -21..-24% against even the degraded
        // bands).
        const double tc_cycles =
            2.0 * (double)q.m * (double)q.n * (double)q.k *
            (double)q.batch /
            ((double)dev.sms * tc_peak(dev.cc, q.perf_class) *
             tc_eff(r.cta));
        return std::max(std::max(l1_cycles, l2_cycles), tc_cycles) /
               wave_eff;
    }
};

// The chain: planners in rank order, first to answer wins. Composition is
// the planner mode — "table" runs the row tiers then the degraded tail,
// "hybrid" inserts the model between them, "model" trusts the model
// alone. Every chain ends in the degraded rows (their bands are open on N
// with -1 keys, and the m=0 degenerate case falls to the first row), so
// dispatch is a total function.
inline PlanDecision plan_dispatch(const PlanQuery& q) {
    static const RowSetPlanner override_planner(
        "override",
        [](const PlanQuery& query) {
            return plan_table_override_source().lookup(query);
        },
        /*respects_table_off=*/true);
    static const RowSetPlanner injected_planner(
        "injected",
        [](const PlanQuery& query) {
            return plan_table_injected_source().lookup(query);
        },
        /*respects_table_off=*/true);
    static const RowSetPlanner builtin_planner(
        "builtin",
        [](const PlanQuery& query) {
            int count = 0;
            const TableRow* rows =
                builtin_plan_table(query.perf_class, count);
            if (rows == nullptr) return std::optional<TableRow>{};
            const TableRow* row = plan_row_for(rows, count, query);
            if (row == nullptr) return std::optional<TableRow>{};
            return std::optional<TableRow>{*row};
        },
        /*respects_table_off=*/true);
    static const RowSetPlanner degraded_planner(
        "degraded",
        [](const PlanQuery& query) {
            // The M band alone decides: the degraded rows are open on N
            // and K with -1 keys and carry no gate. The n of 1 stands for
            // "some real n" — the matcher's band test is "strictly past
            // the min", and an n of 0 sits ON the open bound.
            PlanQuery m_only;
            m_only.m = query.m;
            m_only.n = 1;
            const TableRow* row =
                plan_row_for(kDegradedPlanRows, 3, m_only);
            if (row == nullptr)
                return std::optional<TableRow>{kDegradedPlanRows[0]};
            return std::optional<TableRow>{*row};
        },
        /*respects_table_off=*/false);
    static const ModelPlanner model_planner;

    static constexpr const GemmPlanner* kChainTable[] = {
        &override_planner, &injected_planner, &builtin_planner,
        &degraded_planner};
    static constexpr const GemmPlanner* kChainHybrid[] = {
        &override_planner, &injected_planner, &builtin_planner,
        &model_planner, &degraded_planner};
    static constexpr const GemmPlanner* kChainModel[] = {&model_planner,
                                                         &degraded_planner};

    const int mode = gemm_planner_mode();
    const GemmPlanner* const* chain =
        mode == 2 ? kChainModel : mode == 1 ? kChainHybrid : kChainTable;
    const int chain_len = mode == 2 ? 2 : mode == 1 ? 5 : 4;
    for (int i = 0; i < chain_len; ++i)
        if (std::optional<PlanDecision> d = chain[i]->plan(q)) {
            log_dispatch(q, *d);
            return *d;
        }
    // Unreachable: the degraded planner answers every query (its row
    // function has the m=0 fallback), so the chain above always returns.
    // A release build still needs a value here — the first degraded row
    // is the historical answer for the degenerate shapes.
    const TableRow& row = kDegradedPlanRows[0];
    PlanDecision d{*recipe_of((int)row.cta, row.stages, row.kk, false, 2, 2),
                   0, "degraded"};
    log_dispatch(q, d);
    return d;
}

// The one place a GemmParams becomes planner input, and the one place the
// dispatch key is derived: perf class, operand widths and the crosswise count
// are all functions of the typed call, so they are computed rather than passed
// in and a caller cannot hand the planner a key that contradicts its own types
// and layouts.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB>
PlanQuery plan_query(const GemmParams& p, const DeviceFacts& dev) {
    PlanQuery q;
    q.m = p.m;
    q.n = p.n;
    q.k = p.k;
    q.batch = p.batch;
    q.perf_class = (int)gemm_perf_class<ElemA, ElemB>();
    q.crosswise = crosswise_of<LayoutA, LayoutB>();
    q.ba = (int)sizeof(ElemA);
    q.bb = (int)sizeof(ElemB);
    q.dev = dev;
    return q;
}

// The typed dispatch entry: problem in, decision out. Taking the layout
// tags as types is what ties the decision to the launch that follows it —
// every derived field comes from the same tags the launcher instantiates
// with.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB>
PlanDecision plan_dispatch_for(const GemmParams& p) {
    return plan_dispatch(
        plan_query<ElemA, ElemB, LayoutA, LayoutB>(p, device_facts()));
}
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

// Output-reclaim feasibility of one tile: the epilogue scatters the output
// tile into the reclaimed operand rings, so a fat output (fp32, 4B/elem) can
// outgrow the ring the planner priced — the wide CTA's 128x256 of fp32 is
// 131072B against the 73728B a 1-byte pair leaves it. Keyed on the tile's own
// geometry, which makes this exactly the predicate the launch twins' reclaim
// static_assert states, so a new CTA class or ring depth cannot drift from the
// assert that guards it. One definition serves both dispatch ladders.
template <typename Tile, typename ElemA, typename ElemB, typename OutT>
constexpr bool reclaim_fits() {
    return Tile::CtaShape::kM * Tile::CtaShape::kN * sizeof(OutT) <=
           ring_smem_bytes(Tile::CtaShape::kM, Tile::CtaShape::kN,
                           Tile::CtaShape::kK, Tile::kStages,
                           (int)sizeof(ElemA), (int)sizeof(ElemB));
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
// ring depth, k-tile depth) selects exactly one manifest entry — the ||
// short-circuits — and the resolver maps its tile onto a concrete Policy and
// launches.
template <typename Manifest, typename Resolver>
bool dispatch_tile(const PlanDecision& d, const Resolver& resolve) {
    return std::apply(
        [&d, &resolve](auto... tiles) {
            return (... ||
                    (tile_class<decltype(tiles)>() ==
                         static_cast<TileClass>(d.recipe.cta) &&
                     decltype(tiles)::kStages == d.recipe.stages &&
                     (int)decltype(tiles)::CtaShape::kK == d.recipe.kk &&
                     resolve.template run<decltype(tiles)>()));
        },
        Manifest{});
}

// Narrow twin of a big tile for the output-reclaim fallback, carrying the
// same ring depth as the tile it replaces (the planner priced that ring, so a
// deeper substitute could overflow the smem opt-in). There is no kK=32 narrow
// s3, so a deep kK=32 big tile falls back to its s2 twin.
template <typename Tile>
using narrow_fallback_t = std::conditional_t<
    Tile::CtaShape::kK == 32, Tile_128x64x32_W32x32_S2_Fast,
    std::conditional_t<(Tile::kStages >= 3), Tile_128x64x64_W32x32_S3_Fast,
                       Tile_128x64x64_W32x32_S2_Fast>>;

// TMA ladder resolver. The gate in launch_plan already guarantees
// dual-congruous 1-/2-byte operands, so the fast tile stays; only an
// output-reclaim overflow swaps the CTA for its narrow twin. std::conditional_t
// keeps every alias instantiable, which the kernel's reclaim static_assert
// requires (an if-constexpr branch still NAMES its dead types).
template <typename ElemA, typename ElemB, typename LayoutOut, typename OutT,
          bool UseMx = false>
struct TmaLauncher {
    const GemmParams& p;
    cudaStream_t stream;
    template <typename Tile>
    bool run() const {
        // Only the geometries whose output tile can outgrow their own rings
        // carry a substitution; the narrow and small classes always fit.
        constexpr bool kReclaimGated = tile_class<Tile>() == TileClass::kBig128 ||
                                       tile_class<Tile>() == TileClass::kWide128x256;
        // The small CTA's warp widening, the same rule the cp.async ladder
        // applies (warp_widened_t, policy.cuh).
        using Widened = warp_widened_t<ElemA, ElemB, Tile>;
        using TileT = std::conditional_t<
            kReclaimGated && !reclaim_fits<Widened, ElemA, ElemB, OutT>(),
            narrow_fallback_t<Widened>, Widened>;
        return launch_policy_tma<GemmPolicy<ElemA, ElemB, RowMajor, ColMajor, TileT, LayoutOut,
                       OutT, false, true, UseMx>>(p, stream);
    }
};

// cp.async ladder resolver. Two substitutions, both on the CTA classes whose
// output tile can outgrow their own rings: the fast loop exists only for
// dual-congruous staging (crosswise operands take the predicated generic loop
// — the NonFast twin), and a fat output (fp32, 4B) that cannot reclaim the
// ring routes to the narrow CTA — same math at lower reuse. Narrow and small
// entries pass through.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut, typename OutT, bool kBigFast, bool UseMx = false>
struct CpAsyncLauncher {
    GemmParams p;
    cudaStream_t stream;
    template <typename Tile>
    bool run() const {
        // The small CTA's warp widening: the 8-warp 64x64 twin starves the
        // tensor pipe on two-byte operands (measured 1.36-1.66x for the
        // 16-warp form, parity to -3% on the thinnest shapes). The rule and
        // its arms live in policy.cuh's warp_widened_t.
        using Widened = warp_widened_t<ElemA, ElemB, Tile>;
        using NonFast = GemmTileConfig<typename Widened::CtaShape,
                                       typename Widened::WarpShape,
                                       Widened::kStages, false>;
        constexpr bool kFits = reclaim_fits<Widened, ElemA, ElemB, OutT>();
        constexpr bool kBig = tile_class<Widened>() == TileClass::kBig128;
        constexpr bool kWide = tile_class<Widened>() == TileClass::kWide128x256;
        using TileT = std::conditional_t<
            kBig, std::conditional_t<
                      kFits, std::conditional_t<kBigFast, Widened, NonFast>,
                      narrow_fallback_t<Widened>>,
            std::conditional_t<kWide, std::conditional_t<
                                          kFits, Widened,
                                          narrow_fallback_t<Widened>>,
                               Widened>>;
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
void launch_plan_impl(GemmParams p, const PlanDecision& d,
                      cudaStream_t stream) {
    p.raster = d.raster;
    // Dual-congruous (crosswise 0), the only pair whose plan can reach the
    // fast-loop ladder entries: both operands staged as-is.
    constexpr bool kBigFast = crosswise_of<LayoutA, LayoutB>() == 0;
    // TMA staging first when the layout pair and dtypes allow it (the
    // planner's stage/tile decisions are shared): sm_90+ device, no
    // kill switch, and every descriptor encodable — else the cp.async
    // twin below runs unchanged.
    if constexpr (kBigFast && sizeof(ElemA) <= 2 && sizeof(ElemB) <= 2) {
        if (!gemm_tma_staging_disabled() && astrai::device_facts().cc >= 90 &&
            dispatch_tile<manifest_for<ElemA, ElemB, RowMajor, ColMajor>>(
                d, TmaLauncher<ElemA, ElemB, LayoutOut, OutT, UseMx>{
                        p, stream}))
            return;
    }
    dispatch_tile<manifest_for<ElemA, ElemB, LayoutA, LayoutB>>(
        d, CpAsyncLauncher<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                           kBigFast, UseMx>{p, stream});
}

// The planner entry: symmetric fp8 rides the sm_120 block_scale cell unless
// ASTR_GEMM_NO_MX knocks it out.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16>
void launch_plan(GemmParams p, const PlanDecision& d, cudaStream_t stream) {
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
        if (!gemm_mx_cell_disabled() && astrai::device_facts().cc == 120) {
            launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                             true>(p, d, stream);
            return;
        }
    }
    launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT>(
        p, d, stream);
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
// knobs; the fp8 pairs enter with their element types directly.
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
    // Each branch plans from its OWN layout tags, so the plan and the launch
    // below cannot disagree about the crosswise count, widths or perf class.
    // The tags ride empty tag instances; decltype recovers the types.
    const auto launch = [&](auto la, auto lb, auto lout) {
        launch_plan<ElemA, ElemB, decltype(la), decltype(lb), decltype(lout),
                    OutT>(
            p, plan_dispatch_for<ElemA, ElemB, decltype(la), decltype(lb)>(p),
            stream);
    };
    if (trans_a && trans_b) {
        // The swap computes the transposed problem; its (rewritten TT)
        // branch instantiates the column-major-output epilogue through
        // LayoutOut. Mixed never swaps, so its output stays row-major (and
        // if constexpr keeps the swapped instantiation out of mixed builds).
        if constexpr (kSymmetric) {
            if (swapped)
                launch(ColMajor{}, ColMajor{}, ColMajor{});
            else
                launch(ColMajor{}, ColMajor{}, RowMajor{});
        } else {
            launch(ColMajor{}, ColMajor{}, RowMajor{});
        }
    } else if (trans_b) {
        // NT (the fused-linear shape), the production nn.Linear route.
        launch(RowMajor{}, ColMajor{}, RowMajor{});
    } else if (trans_a) {
        launch(ColMajor{}, RowMajor{}, RowMajor{});
    } else {
        // Dual row-major: mixed only — symmetric NN was rewritten above
        // into the transposed TT kernel (if constexpr keeps this
        // instantiation out of symmetric builds).
        if constexpr (!kSymmetric)
            launch(RowMajor{}, RowMajor{}, RowMajor{});
    }
}

// Host-only planner probe (the Python autotuner's coverage check): the
// decision gemm_dispatch would make for this problem, without a launch —
// the planner is GPU-free by design. The
// tag selection mirrors gemm_dispatch branch-for-branch, symmetric-NN
// rewrite included, so a probe cannot disagree with the branch the real
// call takes; LayoutOut never reaches the planner, so it is absent here.
// The probe returns the decision plus the query it answered (the binding
// reports perf_class/crosswise from the query, source/recipe from the
// decision).
template <typename ElemA, typename ElemB>
std::pair<PlanDecision, PlanQuery> plan_probe_for(
    int64_t m, int64_t n, int64_t k, int64_t batch,
    bool trans_a, bool trans_b, const DeviceFacts& dev) {
    GemmParams p{};  // the planner reads m/n/k/batch only
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.batch = static_cast<int>(batch);
    if constexpr (std::is_same_v<ElemA, ElemB>) {
        canonicalize_gemm(p, trans_a, trans_b);  // symmetric NN -> transposed TT
    }
    auto probe = [&](auto la, auto lb) {
        PlanQuery q =
            plan_query<ElemA, ElemB, decltype(la), decltype(lb)>(p, dev);
        return std::make_pair(plan_dispatch(q), std::move(q));
    };
    if (trans_a && trans_b) return probe(ColMajor{}, ColMajor{});
    if (trans_b) return probe(RowMajor{}, ColMajor{});
    if (trans_a) return probe(ColMajor{}, RowMajor{});
    return probe(RowMajor{}, RowMajor{});
}

}  // namespace gemm
}  // namespace astrai
