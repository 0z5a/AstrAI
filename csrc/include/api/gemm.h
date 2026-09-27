#pragma once
// The gemm module's whole host surface: composed callers inside the module
// (fp8 linear compiles in so the dispatch state stays single-source) and the
// pybind layer (bindings.cu, which owns every py:: spelling in the family).
// Declarations only and deliberately template-free — including it must not
// instantiate the dtype-pair kernels, which are explicitly instantiated in
// their own TUs. No Python types — a py:: signature is callable from nowhere
// but bindings.cu; the flat structs below carry what the Python tooling reads.

#include <c10/util/Optional.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

namespace astrai {
namespace gemm {

// The single quantized-GEMM entry (one kernel for every cell); semantics,
// validation and dispatch are documented at the definition in gemm.cu. The
// pybind ``quant_gemm`` is a thin None-tolerant wrapper over this.
torch::Tensor quant_gemm_impl(torch::Tensor a,
                              torch::Tensor b,
                              c10::optional<torch::Tensor> a_scale,
                              c10::optional<torch::Tensor> b_scale,
                              bool trans_a,
                              bool trans_b,
                              c10::optional<torch::Tensor> bias);

// Planner introspection, GPU-free (launches nothing). `source` names the row
// tier that decided; the rest is the picked recipe in dispatch-key form.
// `crosswise` stays an int (direct-load operand count), read as 0/1 by the
// Python tooling.
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

// Runtime configuration (the backing of astrai.extension.plan). Every field is
// tri-state: absent leaves the knob unchanged, an explicit value wins over the
// one-time env seed; pybind maps an absent dict key to "absent".

// Which row tier a `rows` patch addresses. Tiers rank in this order at lookup
// — override (the experimenter's) beats injected (the autotuner's) beats the
// compiled-in tables.
enum class RowTier : int {
    Override = 0,
    Injected = 1,
};

struct GemmConfigPatch {
    // Row spec: a row-file path, or inline row text (one row per line:
    // m_min m_max n_min n_max perf_class crosswise cta stages raster [kk]).
    // The addressed tier is replaced wholesale; empty string clears it, absent
    // leaves both tiers alone.
    c10::optional<std::string> rows;
    c10::optional<RowTier> tier; // which tier `rows` addresses
    // Every row tier off: rows skipped entirely, the planner chain falls
    // through to its model/degraded end.
    c10::optional<bool> table_off;
    // 0/1/2 = table / hybrid / model; -1 restores unset (the env seed decides;
    // an unseeded process resolves to hybrid).
    c10::optional<int> planner_mode;
    c10::optional<bool> log;
    // Positive enables: staging_tma=false forces cp.async staging,
    // staging_mx=false knocks the sm_120a block-scale cell out (the A/B knobs).
    c10::optional<bool> staging_tma;
    c10::optional<bool> staging_mx;
};

// The whole configuration as a value: configure(patch) returns it, and feeding
// its fields back re-installs exactly this state (each row tier rides the
// source spec it was installed from) — save/restore round-trips in one call.
struct GemmConfigState {
    std::string planner;   // resolved planner mode name
    int planner_mode = -1; // raw knob: -1 = unset (the env seed decides)
    // Installed row counts: a mistyped path that parses to no rows shows up
    // here as 0 rather than staying silent.
    int override_rows = 0;
    int injected_rows = 0;
    std::string override_source; // what that tier was last installed from
    std::string injected_source;

    bool log = false;
    bool table_off = false;
    bool staging_tma = true;
    bool staging_mx = true;
};

// Apply `patch` and return the resulting state in one call — a caller never
// re-reads the knobs it just set.
GemmConfigState configure(const GemmConfigPatch& patch);
GemmConfigState config_state();

// The dispatch's own vocabulary, for the Python tooling and the sweep's C++
// emitter — neither keeps a second copy of the names.

// Every (crosswise, operand widths) ladder's recipes, deduped on the dispatch
// key, in dispatch (manifest) order. A row is (crosswise, ba, bb, cta, stages,
// kk, bm, bn, wm, wn, threads, smem); its numbers spell the canonical name
// Tile_<bm>x<bn>x<kk>_W<wm>x<wn>_S<stages> — how the Python tooling joins rows
// with dataset recipe strings.
std::vector<std::vector<int>> tile_vocabulary();

// The TileClass spellings, in enum order — what a row's cta ordinal expands to
// in the compiled-in tables (the GENERATED block's paste target).
std::vector<const char*> tile_class_names();

} // namespace gemm
} // namespace astrai
