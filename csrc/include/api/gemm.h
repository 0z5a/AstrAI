#pragma once
/*
 * The gemm module's host surface: in-module composed callers (fp8 linear) and
 * the pybind layer (bindings.cu owns every py:: spelling). Declarations only,
 * template-free — including it must not instantiate the dtype-pair kernels,
 * which live in their own TUs; no Python types (py:: is callable from nowhere
 * but bindings.cu).
 */

#include <c10/util/Optional.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

namespace astrai {
namespace gemm {

struct GemmCapabilities {
    int cc;
    bool mma, fp8, tma, mx;
    const char* targets;
};
GemmCapabilities capabilities();

/*
 * The single quantized-GEMM entry (one kernel per cell); semantics, validation
 * and dispatch at the definition in gemm.cu. pybind's quant_gemm is a thin
 * None-tolerant wrapper.
 */
torch::Tensor quant_gemm_impl(torch::Tensor a,
                              torch::Tensor b,
                              c10::optional<torch::Tensor> a_scale,
                              c10::optional<torch::Tensor> b_scale,
                              bool trans_a,
                              bool trans_b,
                              c10::optional<torch::Tensor> bias);

/*
 * Planner introspection, GPU-free. `source` names the deciding row tier, the
 * rest is the recipe in dispatch-key form; `crosswise` stays an int (0/1 for
 * the Python tooling).
 */
struct PlanProbe {
    std::string source;
    int cta = 0;
    int stages = 0;
    int raster = 0;
    int kk = 0;
    int perf_class = -1;
    int crosswise = 0;
};

PlanProbe plan_probe(int64_t m,
                     int64_t n,
                     int64_t k,
                     at::ScalarType dt_a,
                     at::ScalarType dt_b,
                     bool trans_a,
                     bool trans_b,
                     int64_t batch);

/*
 * Runtime configuration (the backing of astrai.extension.policy.gemm.plan). Every field is
 * tri-state: absent leaves the knob unchanged, a value wins over the env seed;
 * pybind maps an absent dict key to "absent".
 */

/*
 * Which row tier a `rows` patch addresses; lookup ranks override > injected >
 * the compiled-in tables.
 */
enum class RowTier : int {
    Override = 0,
    Injected = 1,
};

struct GemmConfigPatch {
    /*
     * Row-file path or inline rows (one per line:
     * m_min m_max n_min n_max perf_class crosswise cta stages raster [kk]);
     * the addressed tier is replaced wholesale, empty string clears it.
     */
    c10::optional<std::string> rows;
    c10::optional<RowTier> tier; // which tier `rows` addresses
    // All row tiers off: the planner chain falls through to model/degraded.
    c10::optional<bool> table_off;
    /*
     * 0/1/2 = table / hybrid / model; -1 = unset (env seed decides, hybrid
     * unseeded).
     */
    c10::optional<int> planner_mode;
    c10::optional<bool> log;
    /*
     * Positive enables: false forces cp.async staging / knocks the sm_120a
     * block-scale cell out (the A/B knobs).
     */
    c10::optional<bool> staging_tma;
    c10::optional<bool> staging_mx;
};

/*
 * configure(patch) returns the resulting state; feeding it back re-installs it
 * exactly — save/restore round-trips in one call.
 */
struct GemmConfigState {
    std::string planner;   // resolved planner mode name
    int planner_mode = -1; // -1 = unset (env seed decides)
    /*
     * Installed row counts: 0 means the source parsed to no rows (e.g. a
     * mistyped path) rather than staying silent.
     */
    int override_rows = 0;
    int injected_rows = 0;
    std::string override_source; // what that tier was installed from
    std::string injected_source;

    bool log = false;
    bool table_off = false;
    bool staging_tma = true;
    bool staging_mx = true;
};

// Apply `patch` and return the resulting state in one call.
GemmConfigState configure(const GemmConfigPatch& patch);
GemmConfigState config_state();

/*
 * The dispatch vocabulary, for the Python tooling and the sweep's C++ emitter
 * — neither keeps a second copy of the names.
 */

/*
 * Every ladder's recipes, deduped on the dispatch key, in dispatch order. A
 * row is (crosswise, ba, bb, cta, stages, kk, bm, bn, wm, wn, threads, smem);
 * its numbers spell the canonical name
 * Tile_<bm>x<bn>x<kk>_W<wm>x<wn>_S<stages> — the Python tooling's join key.
 */
std::vector<std::vector<int>> tile_vocabulary();

/*
 * TileClass spellings in enum order — a row's cta ordinal expands to these in
 * the compiled-in tables.
 */
std::vector<const char*> tile_class_names();

} // namespace gemm
} // namespace astrai
