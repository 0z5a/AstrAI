#pragma once
// AOT dispatch table: measured best-recipe rows per (shape band, dtype
// class, layout class) that plan_gemm consults. Rows are data — the launch
// ladders resolve a row's CTA class through the manifest, so this header
// holds no kernel pointers or registration. Sources, override first:
//   - a runtime table file (ASTR_GEMM_TABLE=/path/to/rows.txt), so tile
//     tuning never needs a rebuild;
//   - the compiled-in GENERATED rows below (paste the row file the
//     measurement script emits; the script never writes source).
// An empty table makes every lookup miss and dispatch falls through to the
// degraded band rows.
// The sweep times the fused-linear (NT) layout, so pasted rows carry
// crosswise 0: non-NT shapes (TT, TN, mixed dual-row-major) take the same
// degraded fallback.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "policy.cuh"

namespace astrai {
namespace gemm {

// Ring K a row carries when its file omits the field, and the depth the
// forced-recipe knob pins. The k-tile depth is a row field now that the
// manifest holds kK twins (policy.cuh): the measured winner is kK=32 on most
// shapes, but not on all, so it has to be per row.
static constexpr int kTableRowK = 64;

// The k-tile depths the manifests actually carry as tiles. A row naming any
// other depth would match no tile and launch nothing, so plan_from_row
// rejects it and the next source (or the degraded bands) serves the shape.
inline constexpr bool row_k_supported(int kk) { return kk == 32 || kk == 64; }

// One tuned row. Bands are (min, max] on M and N — min exclusive,
// max inclusive, 0 = unbounded (humming's dispatch-table convention);
// a row matches when m > m_min && (m_max == 0 || m <= m_max) and the
// same for n. perf_class is the GemmPerfClass id (0 W16A16 / 1 W8A16 /
// 2 W8A8 / 3 F8A8; -1 = any): the same band can price a different
// recipe per dtype class. crosswise is the crosswise-operand count of
// the problem (0 = dual-congruous NT, 1 = TT, the NN swap and the mixed
// dual-row-major NN case, 2 = TN — the
// (trans_a ? 1 : 0) + (trans_b ? 0 : 1) of gemm_dispatch; -1 = any).
// raster: 0 = plan_raster with this row's CTA geometry at launch time.
// kk: the row's ring K; a row file that omits the trailing field keeps
// kTableRowK.
struct TableRow {
    int64_t m_min;
    int64_t m_max;
    int64_t n_min;
    int64_t n_max;
    int perf_class;
    int crosswise;
    TileClass cta;
    int stages;
    int raster;
    int kk = kTableRowK;
};

// CTA geometry of one class — the shapes dispatch_tile resolves from the
// manifest, so a row's smem budget can be priced here.
inline constexpr void plan_row_geometry(TileClass cta, int& bm, int& bn) {
    if (cta == TileClass::kWide128x256) bm = 128, bn = 256;
    else if (cta == TileClass::kBig128) bm = 128, bn = 128;
    else if (cta == TileClass::kNarrow128x64) bm = 128, bn = 64;
    else  bm = 64, bn = 64;
}

// First matching row wins (generated tables are emitted non-overlapping;
// the override file can shadow the builtin rows when it precedes them).
inline const TableRow* plan_row_for(const TableRow* rows, int count,
                                    int64_t m, int64_t n, int perf_class,
                                    int crosswise) {
    for (int i = 0; i < count; ++i) {
        const TableRow& r = rows[i];
        if (m <= r.m_min) continue;
        if (r.m_max != 0 && m > r.m_max) continue;
        if (n <= r.n_min) continue;
        if (r.n_max != 0 && n > r.n_max) continue;
        if (r.perf_class != -1 && r.perf_class != perf_class) continue;
        if (r.crosswise != -1 && r.crosswise != crosswise) continue;
        return &r;
    }
    return nullptr;
}

// Row file format: one row per line, whitespace-separated
//   m_min m_max n_min n_max perf_class crosswise cta stages raster
// ('#' starts a comment; 'perf_class' 0..3 / -1 any; 'crosswise'
// 0..2 / -1 any; 'cta' is 0 small / 1 narrow / 2 big — the TileClass
// order; 'stages' 2..3 (the s4/s5 deep rings measured no gain on the TMA
// ring — two stages hit the latency floor, deeper stages only cost smem
// residency; see policy.cuh). Invalid lines are warn-and-skip: tuning
// files are hand-edited between sweeps, and a malformed row must never
// block a launch the fallback would serve.
inline bool parse_plan_table_file(const std::string& path,
                                  std::vector<TableRow>& rows) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return false;
    char line[256];
    int lineno = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        ++lineno;
        if (char* hash = std::strchr(line, '#'); hash != nullptr) *hash = '\0';
        long long m_min, m_max, n_min, n_max;
        int perf_class, crosswise, cta, stages, raster, kk;
        int got =
            std::sscanf(line, " %lld %lld %lld %lld %d %d %d %d %d %d", &m_min,
                        &m_max, &n_min, &n_max, &perf_class, &crosswise, &cta,
                        &stages, &raster, &kk);
        if (got == EOF) continue;
        if (got == 9) {
            kk = kTableRowK;  // a 9-field row predates the k field
            got = 10;         // and satisfies the validation below
        }
        if (got != 10 || m_min < 0 || n_min < 0 || !row_k_supported(kk) ||
            (m_max != 0 && m_max < m_min) ||
            (n_max != 0 && n_max < n_min) || perf_class < -1 || perf_class > 3 ||
            crosswise < -1 || crosswise > 2 || cta < 0 || cta > 3 ||
            (stages != 2 && stages != 3)) {
            std::fprintf(stderr,
                         "[gemm-plan-table] %s:%d: ignoring malformed row\n",
                         path.c_str(), lineno);
            continue;
        }
        rows.push_back({m_min, m_max, n_min, n_max, perf_class, crosswise,
                        cta == 3   ? TileClass::kWide128x256
                        : cta == 2 ? TileClass::kBig128
                        : cta == 1 ? TileClass::kNarrow128x64
                                   : TileClass::kSmall64,
                        stages, raster, kk});
    }
    std::fclose(f);
    return true;
}

// Cache of the override file, re-parsed only when the env path changes
// (a single setenv per process in practice; the parse result is
// idempotent, so a concurrent writer races benignly like device_facts).
inline const std::vector<TableRow>& plan_table_override_rows() {
    static std::vector<TableRow> rows;
    static std::string loaded_path;
    const char* env = std::getenv("ASTR_GEMM_TABLE");
    const std::string path = env != nullptr ? std::string(env) : std::string();
    if (path != loaded_path) {
        rows.clear();
        if (!path.empty() && path != "-") parse_plan_table_file(path, rows);
        loaded_path = path;
    }
    return rows;
}

// BEGIN GENERATED
// Measured power-of-2 grid table (2026-09-10): the band-search partition of
// a sweep over M, N, K in 32..4096 powers of two, all seven dtype combos /
// four perf classes (gen_plan_table.py --full-coverage --band-search),
// compressed 42 -> 14 rows by compress_plan_table.py — abutting same-recipe
// rectangles merged, the catch-alls the open last N band already shadows
// dropped, verified decision-identical over 80656 probe points x 4 classes
// x 2 crosswise counts, so the compression costs nothing at dispatch.
// The distillate these rows replace keyed the recipe on one N split at
// 1280; the measured recipe depends on N far more strongly than that for
// large M. It sent M>2560, N>1280 to the narrow CTA where the big CTA is
// 1.40x faster (4096x4096x4096 w16a16 101 -> 142 TFLOPS), and M>2560,
// N<=1280 to the big CTA where the small CTA is up to 3.6x faster
// (4096x64x4096 16.5 -> 57) — a 128-wide N tile wastes half its mma on a
// 64-column problem; the quantized classes had no rows at all and fell to
// the degraded bands. Measured on the 42-row form (validate_plan_table.py,
// interleaved A/B, 26 holdout shapes x 6 combos): grid 1.199x, LLM shape
// list 1.097x, combined 1.109x; worst per-shape regression 0.84x.
static constexpr TableRow kBuiltinPlanTable[] = {
    {0, 0, 0, 768, 0, 0, TileClass::kSmall64, 3, 0},
    {0, 768, 768, 1536, 0, 0, TileClass::kSmall64, 3, 0},
    {768, 0, 768, 1536, 0, 0, TileClass::kBig128, 2, 0},
    {0, 384, 1536, 3072, 0, 0, TileClass::kSmall64, 3, 0},
    {384, 0, 1536, 0, 0, 0, TileClass::kBig128, 2, 0},
    {0, 96, 3072, 0, 0, 0, TileClass::kSmall64, 3, 0},
    {96, 384, 3072, 0, 0, 0, TileClass::kNarrow128x64, 2, 0},
    {0, 0, 0, 1536, 1, 0, TileClass::kSmall64, 3, 0},
    {0, 768, 1536, 3072, 1, 0, TileClass::kSmall64, 3, 0},
    {768, 0, 1536, 3072, 1, 0, TileClass::kBig128, 2, 0},
    {0, 384, 3072, 0, 1, 0, TileClass::kSmall64, 3, 0},
    {384, 0, 3072, 0, 1, 0, TileClass::kBig128, 2, 0},
    {0, 0, 0, 0, 2, 0, TileClass::kSmall64, 3, 0},
    {0, 0, 0, 0, 3, 0, TileClass::kSmall64, 3, 0},
};
// END GENERATED
static constexpr int kBuiltinPlanTableCount =
    (int)(sizeof(kBuiltinPlanTable) / sizeof(TableRow));

// Override file first, then the compiled-in rows. ASTR_GEMM_TABLE="-"
// is the explicit "AOT off" escape hatch: neither override nor builtin
// rows, so dispatch falls through to the degraded band rows (dev/bench).
inline const TableRow* plan_table_lookup(const GemmParams& p, int perf_class,
                                         int crosswise) {
    const char* env = std::getenv("ASTR_GEMM_TABLE");
    if (env != nullptr && std::strcmp(env, "-") == 0) return nullptr;
    const std::vector<TableRow>& rows = plan_table_override_rows();
    if (const TableRow* row = plan_row_for(rows.data(), (int)rows.size(), p.m,
                                           p.n, perf_class, crosswise);
        row != nullptr)
        return row;
    return plan_row_for(kBuiltinPlanTable, kBuiltinPlanTableCount, p.m, p.n,
                        perf_class, crosswise);
}

// One dispatch source: name is the ASTR_GEMM_PLAN decision tag, and find
// yields the source's first row matching the problem (nullopt = this
// source has no say). plan_gemm's precedence is the source array order.
struct TableSource {
    const char* name;
    std::optional<TableRow> (*find)(const GemmParams& p, int perf, int crosswise);
};

// ASTR_GEMM_RECIPE=big|narrow|small: one synthetic open row (s2 ring,
// raster auto) forcing the CTA class — the calibration knob the table's
// rows are measured with. An unknown value yields no row. The env is re-
// read on every call (no per-process cache): the sweep interleaves the
// candidates at each shape in a single process, so the comparison shares
// one GPU clock/thermal state.
inline std::optional<TableRow> env_recipe_row(const GemmParams&, int, int) {
    const char* name = std::getenv("ASTR_GEMM_RECIPE");
    if (name == nullptr) return std::nullopt;
    TileClass cta;
    if (std::strcmp(name, "big") == 0)
        cta = TileClass::kBig128;
    else if (std::strcmp(name, "narrow") == 0)
        cta = TileClass::kNarrow128x64;
    else if (std::strcmp(name, "small") == 0)
        cta = TileClass::kSmall64;
    else
        return std::nullopt;
    return TableRow{0, 0, 0, 0, -1, -1, cta, 2, 0};
}

// The AOT table source: override file first, then the compiled-in rows.
inline std::optional<TableRow> table_row(const GemmParams& p, int perf,
                                         int crosswise) {
    if (const TableRow* row = plan_table_lookup(p, perf, crosswise);
        row != nullptr)
        return *row;
    return std::nullopt;
}

// Last-resort rows for a table miss with the model retired: the M band's
// dominant recipe from the full-coverage sweep (small for short M, narrow
// mid, big past mid) — a safe default, never best. Open N with -1 keys
// matches every shape, so planning stays a total function.
static constexpr TableRow kDegradedPlanRows[] = {
    {0, 512, 0, 0, -1, -1, TileClass::kSmall64, 2, 0},
    {512, 3072, 0, 0, -1, -1, TileClass::kNarrow128x64, 2, 0},
    {3072, 0, 0, 0, -1, -1, TileClass::kBig128, 2, 0},
};

inline const TableRow& degraded_row_for(int64_t m) {
    if (const TableRow* row =
            plan_row_for(kDegradedPlanRows, 3, m, /*n=*/1, -1, -1);
        row != nullptr)
        return *row;
    return kDegradedPlanRows[0];  // the degenerate m=0 matches no band
}

}  // namespace gemm
}  // namespace astrai
