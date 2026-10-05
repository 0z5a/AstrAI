/*
 * GEMM family, typed host layer (module `gemm`): the dtype-pair registry, its
 * two lookups, and the planner's C++ face — probe, row injection, runtime
 * configuration and the tile vocabulary, all declared in api/gemm.h.
 * quant_gemm's op-entry ladder lives next door in entry.h, and this TU's
 * quant_gemm_impl (bottom) is the wrapper that instantiates it with the
 * dtype-pair lookup defined above; the lookup rides in as a template
 * parameter, so the header needs no include-order contract.
 * The pybind surface (argument marshalling, the dict shapes, the module
 * registration) lives in bindings.cu; this TU holds no py:: type.
 * torch/extension.h is here for the torch::Tensor spelling only.
 *
 * The policy instantiation space compiles one explicit instantiation per dtype
 * pair (one per .cu below), so the heavy template work runs as parallel
 * nvcc jobs. This TU keeps the dtype-pair switch; the C tests instantiate from
 * the headers instead.
 */

#include <c10/core/ScalarType.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "entry.h"
#include <api/dtype.h>
#include <api/gemm.h>
#include <gemm_build.h>
#include <launcher/gemm_dispatch.cuh>
#include <launcher/plan_table.h>
#include <launcher/planning.h>

using namespace astrai;
using namespace astrai::quant;

namespace astrai {
namespace gemm {

#define ASTRAI_GEMM_BASE_PAIRS(X)                                                              \
    X(bf16, bf16)                                                                              \
    X(bf16, int8_t)                                                                            \
    X(int8_t, int8_t)

#if ASTRAI_BUILD_FP8
#define ASTRAI_GEMM_FP8_PAIRS(X)                                                               \
    X(fp8_e4m3, fp8_e4m3)                                                                      \
    X(fp8_e5m2, fp8_e5m2)
#else
#define ASTRAI_GEMM_FP8_PAIRS(X)
#endif
#define ASTRAI_GEMM_PAIRS(X) ASTRAI_GEMM_BASE_PAIRS(X) ASTRAI_GEMM_FP8_PAIRS(X)

#define GEMM_EXTERN(TA, TB)                                                                    \
    extern template void gemm_dispatch<TA, TB, __nv_bfloat16, MmaSync>(                        \
        GemmParams, cudaStream_t, bool, bool);
ASTRAI_GEMM_PAIRS(GEMM_EXTERN)
#undef GEMM_EXTERN
#if ASTRAI_BUILD_TMA
#define GEMM_EXTERN(TA, TB)                                                                    \
    extern template void gemm_dispatch<TA, TB, __nv_bfloat16, TmaMma>(                         \
        GemmParams, cudaStream_t, bool, bool);
ASTRAI_GEMM_PAIRS(GEMM_EXTERN)
#undef GEMM_EXTERN
#endif
#if ASTRAI_BUILD_MX
#define GEMM_EXTERN(TA, TB)                                                                    \
    extern template void gemm_dispatch<TA, TB, __nv_bfloat16, Sm120Mma>(                       \
        GemmParams, cudaStream_t, bool, bool);
ASTRAI_GEMM_FP8_PAIRS(GEMM_EXTERN)
#undef GEMM_EXTERN
#endif

GemmCapabilities capabilities() {
    const int cc = device_facts().cc;
    return {cc, supports(build::kBase, cc), supports(build::kFp8, cc),
            supports(build::kTma, cc), supports(build::kMx, cc), build::kTargets};
}

namespace {

/*
 * The lookup key both stamped switches below pack their case labels with:
 * one u16 per pair, so every label is a compile-time constant and the switch
 * lowers to one indexed branch with no runtime-initialized state. An
 * unsupported pair raises with the actual operand dtypes in the message
 * instead of a hardcoded list that can drift.
 */
using GemmDispatchFn = void (*)(GemmParams, cudaStream_t, bool, bool);

constexpr uint16_t pack_dtypes(c10::ScalarType a, c10::ScalarType b) {
    return static_cast<uint16_t>(static_cast<uint8_t>(a)) << 8 | static_cast<uint8_t>(b);
}

/*
 * The unsupported-pair arm, one spelling for the two lookups below: the
 * error text is generated from ASTRAI_GEMM_PAIRS, matching both lookups.
 */
std::string unsupported_pair_message(c10::ScalarType a, c10::ScalarType b) {
    std::string instantiated;
#define ASTRAI_GEMM_PAIR_ROW(TA, TB)                                                           \
    instantiated += std::string(instantiated.empty() ? "" : ", ") +                            \
                    toString(scalar_type_v<TA>) + " x " + toString(scalar_type_v<TB>);
    ASTRAI_GEMM_PAIRS(ASTRAI_GEMM_PAIR_ROW)
#undef ASTRAI_GEMM_PAIR_ROW
    return std::string("unsupported operand dtype pair ") + toString(a) + " x " +
           toString(b) + " (instantiated: " + instantiated + ")";
}

/* All kernel variants have distinct schedule types and architecture images. */
template <typename A, typename B, typename F>
auto with_schedule(F&& fn) {
    constexpr bool fp8 = sizeof(A) == 1 && !std::is_same_v<A, int8_t>;
    const GemmCapabilities caps = capabilities();
    TORCH_CHECK(fp8 ? caps.fp8 : caps.mma,
                "GEMM kernels were not built for this device/dtype (SM", caps.cc,
                ", compiled targets: ", caps.targets, ")");
#if ASTRAI_BUILD_MX
    if constexpr (fp8) {
        if (caps.mx && !gemm_mx_cell_disabled())
            return fn(Sm120Mma{});
    }
#endif
#if ASTRAI_BUILD_TMA
    if (caps.tma && !gemm_tma_staging_disabled())
        return fn(TmaMma{});
#endif
    return fn(MmaSync{});
}

template <typename A, typename B>
GemmDispatchFn selected_dispatch() {
    return with_schedule<A, B>([](auto schedule) -> GemmDispatchFn {
        return &gemm_dispatch<A, B, __nv_bfloat16, decltype(schedule)>;
    });
}

GemmDispatchFn find_gemm_dispatch(c10::ScalarType a, c10::ScalarType b) {
#define GEMM_CASE(TA, TB)                                                                      \
    case pack_dtypes(scalar_type_v<TA>, scalar_type_v<TB>):                                    \
        return selected_dispatch<TA, TB>();
    switch (pack_dtypes(a, b)) {
    ASTRAI_GEMM_PAIRS(GEMM_CASE)
    }
    throw std::runtime_error(unsupported_pair_message(a, b));
#undef GEMM_CASE
}

/*
 * Host-only functions that run the planner without a launch (the
 * autotuner's coverage check).
 */
using GemmProbeFn = std::pair<PlanDecision, PlanQuery> (*)(
    int64_t, int64_t, int64_t, int64_t, bool, bool, const DeviceFacts&);

template <typename A, typename B>
GemmProbeFn selected_probe() {
    return with_schedule<A, B>([](auto schedule) -> GemmProbeFn {
        return &plan_probe_for<A, B, decltype(schedule)>;
    });
}

GemmProbeFn find_gemm_probe(c10::ScalarType a, c10::ScalarType b) {
#define PROBE_CASE(TA, TB)                                                                     \
    case pack_dtypes(scalar_type_v<TA>, scalar_type_v<TB>):                                    \
        return selected_probe<TA, TB>();
    switch (pack_dtypes(a, b)) {
    ASTRAI_GEMM_PAIRS(PROBE_CASE)
    }
    throw std::runtime_error(unsupported_pair_message(a, b));
#undef PROBE_CASE
}

} // namespace

/*
 * Planner introspection: the Python tooling's C++ face. The planner is
 * GPU-free by design, so the probe launches nothing. Rows reach the planner
 * through configure()'s rows channel; injected rows rank BELOW the override
 * rows (plan_table.h), keeping the override tier authoritative.
 */

PlanProbe plan_probe(int64_t m,
                     int64_t n,
                     int64_t k,
                     at::ScalarType dt_a,
                     at::ScalarType dt_b,
                     bool trans_a,
                     bool trans_b,
                     int64_t batch) {
    const auto [decision, query] =
        find_gemm_probe(dt_a, dt_b)(m, n, k, batch, trans_a, trans_b, astrai::device_facts());
    PlanProbe r;
    r.source = decision.source;
    r.cta = decision.recipe.cta;
    r.stages = decision.recipe.stages;
    r.raster = decision.raster;
    r.kk = decision.recipe.kk;
    r.perf_class = query.perf_class;
    r.crosswise = query.crosswise;
    return r;
}

namespace {

/*
 * Install one row tier from a spec (a row-file path when one opens, else
 * inline row text) and remember the spec, so the config state can hand back a
 * value that re-installs it. `label` is what a parse error reports.
 */
int install_rows(RowSource& tier, const char* label, const std::string& source) {
    std::vector<TableRow> rows;
    if (!parse_plan_table_file(source, rows))
        parse_plan_table_text(source, label, rows);
    const int installed = (int)rows.size();
    tier.set_from(source, std::move(rows));
    return installed;
}

RowSource& row_tier(RowTier tier) {
    return tier == RowTier::Injected ? plan_table_injected_source() : plan_table_override_source();
}

} // namespace

/*
 * Runtime configuration: the backing of astrai.extension.policy.gemm.plan. Every knob is
 * tri-state — an absent patch field leaves it unchanged, an explicit value wins
 * over the one-time env seed. Rows are addressed by tier (`rows` + `tier`), the
 * all-tiers-off switch is its own field, and the staging keys are positive
 * enables: tma=false forces cp.async staging, mx=false knocks the sm_120a
 * block-scale cell out (the A/B knobs).
 */

GemmConfigState config_state() {
    GemmConfigState s;
    s.planner = kPlannerModeNames[gemm_planner_mode()]; // resolves unset
    s.planner_mode = gemm_config().planner.load(std::memory_order_relaxed);
    s.log = gemm_plan_log_enabled();
    s.table_off = gemm_table_off();
    s.override_rows = (int)plan_table_override_source().size();
    s.override_source = plan_table_override_source().source();
    s.injected_rows = (int)plan_table_injected_source().size();
    s.injected_source = plan_table_injected_source().source();
    s.staging_tma = !gemm_tma_staging_disabled();
    s.staging_mx = !gemm_mx_cell_disabled();
    return s;
}

GemmConfigState configure(const GemmConfigPatch& patch) {
    gemm_config_seed_once();
    if (patch.planner_mode.has_value()) {
        const int mode = *patch.planner_mode;
        if (mode < -1 || mode >= kPlannerModeCount)
            throw std::invalid_argument("planner mode must be -1..2");
        gemm_config().planner = mode;
    }
    if (patch.log.has_value())
        gemm_config().log = *patch.log ? 1 : 0;
    if (patch.staging_tma.has_value())
        gemm_config().tma_disabled = *patch.staging_tma ? 0 : 1;
    if (patch.staging_mx.has_value())
        gemm_config().mx_disabled = *patch.staging_mx ? 0 : 1;
    if (patch.table_off.has_value())
        gemm_config().table_off = *patch.table_off ? 1 : 0;
    if (patch.rows.has_value()) {
        const RowTier which = patch.tier.value_or(RowTier::Override);
        if (patch.rows->empty()) {
            row_tier(which).clear();
        } else {
            install_rows(row_tier(which),
                         which == RowTier::Injected ? "injected rows" : "override rows",
                         *patch.rows);
        }
    }
    return config_state();
}

/*
 * The recipe vocabulary per (crosswise, operand widths) — every
 * (CTA class, stages, kK) the launch ladders instantiate for that staging
 * pair, deduped on the dispatch key, in dispatch (manifest) order. Rows are
 * (crosswise, ba, bb, cta, stages, kk, bm, bn, wm, wn, threads, smem);
 * a row's numbers spell its canonical name,
 * Tile_<bm>x<bn>x<kk>_W<wm>x<wn>_S<stages> — the bench's own spelling —
 * which is how the Python tooling joins rows with dataset recipe strings
 * without keeping a second copy of the vocabulary. (2,1) covers the mixed
 * W8A16 class, whose congruent staging runs the conservative
 * ladder too (manifest_kind's fallback); (1,2) matches no supported pair.
 */
std::vector<std::vector<int>> tile_vocabulary() {
    const std::pair<int, int> widths[] = {{2, 2}, {2, 1}, {1, 1}};
    std::vector<std::vector<int>> out;
    for (int crosswise = 0; crosswise <= 1; ++crosswise)
        for (const auto& [ba, bb] : widths)
            for (const GemmRecipe& r : gemm_recipes_for(crosswise != 0, ba, bb))
                out.push_back({crosswise, ba, bb, r.cta, r.stages, r.kk, r.bm, r.bn, r.wm, r.wn,
                               r.threads, r.smem});
    return out;
}

/*
 * The TileClass spellings, in enum order — what a row's cta ordinal expands
 * to in the compiled-in tables (the GENERATED block's paste target). Owned
 * here so the sweep's C++ emitter needs no Python-side copy of the names.
 */
std::vector<const char*> tile_class_names() {
    static constexpr const char* kNames[] = {"kSmall64", "kNarrow128x64", "kBig128", "kWide128x256",
                                             "kTall64x128"};
    static_assert((int)TileClass::kTall64x128 == (int)(sizeof(kNames) / sizeof(kNames[0])) - 1,
                  "kNames is indexed by TileClass: keep it in enum order");
    return std::vector<const char*>(kNames, kNames + sizeof(kNames) / sizeof(kNames[0]));
}

/*
 * quant_gemm's op entry: the ladder itself lives in entry.h (one kernel for
 * every cell, the only kernel-facing export); this wrapper is the one place
 * the family's dtype-pair lookup meets it — as a template argument, so the
 * header needs neither a forward declaration of a TU-internal symbol nor an
 * include-order contract. The scale contract is stated in the header and in
 * docs/developer/kernels/gemm.md, "Scales".
 */
torch::Tensor quant_gemm_impl(torch::Tensor a,
                              torch::Tensor b,
                              c10::optional<torch::Tensor> a_scale,
                              c10::optional<torch::Tensor> b_scale,
                              bool trans_a,
                              bool trans_b,
                              c10::optional<torch::Tensor> bias) {
    return quant_gemm_ladder<&find_gemm_dispatch>(a, b, a_scale, b_scale, trans_a, trans_b, bias);
}

} // namespace gemm
} // namespace astrai
