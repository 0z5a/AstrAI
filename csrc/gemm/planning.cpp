/* GEMM host planning: recipe enumeration, row selection, and model ranking. */
#include <launcher/planning.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <launcher/plan_table.h>
#include <policy/manifest.cuh>

namespace astrai {
namespace gemm {

// Read legacy ASTR_GEMM_* defaults once; configure() can override them.
void gemm_config_seed_once() {
    static const bool seeded = [] {
        GemmConfig& c = gemm_config();
        auto env = [](const char* name) {
            const char* e = std::getenv(name);
            return e == nullptr ? std::string() : std::string(e);
        };
        if (const std::string v = env("ASTR_GEMM_MODEL"); !v.empty())
            c.planner = std::atoi(v.c_str());
        if (const std::string v = env("ASTR_GEMM_PLAN"); !v.empty() && v != "0")
            c.log = 1;
        if (env("ASTR_GEMM_NO_TMA") == "1")
            c.tma_disabled = 1;
        if (env("ASTR_GEMM_NO_MX") == "1")
            c.mx_disabled = 1;
        if (const std::string v = env("ASTR_GEMM_TABLE"); !v.empty()) {
            if (v == "-") {
                c.table_off = 1;
            } else {
                std::vector<TableRow> rows;
                if (parse_plan_table_file(v, rows))
                    plan_table_override_source().set_from(v, std::move(rows));
            }
        }
        return true;
    }();
    (void)seeded;
}

// Raster the longer grid dimension; keep M-side operand tiles inside the L2 budget.
int plan_raster(const PlanQuery& q, int bm, int bn) {
    const int64_t m_tiles = (q.m + bm - 1) / bm;
    const int64_t n_tiles = (q.n + bn - 1) / bn;
    if (m_tiles < n_tiles)
        return -8;
    if (q.n * q.k * (int64_t)q.bb <= q.dev.l2_bytes * 7 / 10)
        return 1;
    const double reserve = 0.12 + 0.28 * (double)q.bb / (double)q.ba;
    const double budget = (1.0 - std::min(reserve, 0.5)) * (double)q.dev.l2_bytes;
    const int64_t ub = (int64_t)(budget / ((double)bm * (double)q.k * q.ba));
    const int64_t lb = (q.dev.sms + n_tiles - 1) / n_tiles;
    int64_t g = std::min(ub, m_tiles);
    if (ub >= lb)
        g = std::min(std::max(g, lb), m_tiles);
    return (int)std::max(g, (int64_t)1);
}

// Build the same recipe for introspection and model scans.
namespace {

template <typename Tile> inline GemmRecipe recipe_for_tile(int ba, int bb) {
    return GemmRecipe{(int)tile_class<Tile>(),
                      Tile::kStages,
                      (int)Tile::CtaShape::kK,
                      Tile::CtaShape::kM,
                      Tile::CtaShape::kN,
                      Tile::WarpShape::kM,
                      Tile::WarpShape::kN,
                      (Tile::CtaShape::kM / Tile::WarpShape::kM) *
                          (Tile::CtaShape::kN / Tile::WarpShape::kN) * 32,
                      ring_smem_bytes(Tile::CtaShape::kM, Tile::CtaShape::kN, Tile::CtaShape::kK,
                                      Tile::kStages, ba, bb)};
}

template <typename Tile> constexpr int recipe_key() {
    return (int)tile_class<Tile>() | (Tile::kStages << 8) | ((int)Tile::CtaShape::kK << 16);
}

template <typename Manifest> constexpr bool unique_recipe_keys() {
    std::array<int, std::tuple_size_v<Manifest>> keys{};
    std::size_t count = 0;
    std::apply([&](auto... tiles) { ((keys[count++] = recipe_key<decltype(tiles)>()), ...); },
               Manifest{});
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = i + 1; j < count; ++j)
            if (keys[i] == keys[j])
                return false;
    return true;
}

template <typename Manifest, typename F>
inline void for_each_recipe(int ba, int bb, F&& fn) {
    static_assert(unique_recipe_keys<Manifest>(), "manifest dispatch keys must be unique");
    std::apply([&](auto... tiles) { (fn(recipe_for_tile<decltype(tiles)>(ba, bb)), ...); },
               Manifest{});
}

// Match the manifests compiled by the launch ladder.
template <typename F> inline auto with_manifest(bool crosswise_staging, int ba, int bb, F&& fn) {
    switch (manifest_kind(crosswise_staging, ba, bb)) {
    case ManifestKind::kTwoByte:
    case ManifestKind::kMixed:
        return fn(TileManifest{});
    case ManifestKind::kByte:
        return fn(TileManifestByte{});
    default:
        return fn(TileManifestCross{});
    }
}

} // namespace

// Every recipe the ladders instantiate for one staging pair.
std::vector<GemmRecipe> gemm_recipes_for(bool crosswise_staging, int ba, int bb) {
    std::vector<GemmRecipe> out;
    with_manifest(crosswise_staging, ba, bb, [&](auto manifest) {
        for_each_recipe<decltype(manifest)>(ba, bb,
                                            [&](GemmRecipe recipe) { out.push_back(recipe); });
    });
    return out;
}

// A row is eligible only if its exact (class, stages, K) recipe was instantiated.
namespace {

template <typename Manifest>
inline bool recipe_scan(int cta, int stages, int kk, int ba, int bb, GemmRecipe& out) {
    bool found = false;
    auto consider = [&](auto tile) {
        using T = decltype(tile);
        if (found)
            return;
        if ((int)tile_class<T>() != cta || (int)T::kStages != stages || (int)T::CtaShape::kK != kk)
            return;
        out = recipe_for_tile<T>(ba, bb);
        found = true;
    };
    std::apply([&](auto... tiles) { (consider(tiles), ...); }, Manifest{});
    return found;
}

inline std::optional<GemmRecipe>
recipe_of(int cta, int stages, int kk, bool crosswise, int ba, int bb) {
    GemmRecipe out{};
    const bool found = with_manifest(crosswise, ba, bb, [&](auto manifest) {
        return recipe_scan<decltype(manifest)>(cta, stages, kk, ba, bb, out);
    });
    if (!found)
        return std::nullopt;
    return out;
}

// The [gemm-plan] decision line (gen_plan_table's tag regex reads it).
inline void log_dispatch(const PlanQuery& q, const PlanDecision& d) {
    if (!gemm_plan_log_enabled())
        return;
    std::fprintf(stderr,
                 "[gemm-plan] %s m%lld n%lld k%lld b=%d -> cta%d s%d "
                 "raster %d\n",
                 d.source, (long long)q.m, (long long)q.n, (long long)q.k, (int)q.batch,
                 d.recipe.cta, d.recipe.stages, d.raster);
}

// Validate table recipes against the compiled manifest and device limits.
std::optional<PlanDecision> row_plan(const PlanQuery& q, std::optional<TableRow> row,
                                     const char* source) {
    if (!row)
        return std::nullopt;
    const auto recipe = recipe_of((int)row->cta, row->stages, row->kk,
                                  q.crosswise > 0, q.ba, q.bb);
    if (!recipe || recipe->smem > q.dev.smem_max)
        return std::nullopt;
    return PlanDecision{*recipe,
                        row->raster != 0 ? row->raster : plan_raster(q, recipe->bm, recipe->bn),
                        source};
}

std::optional<TableRow> builtin_row(const PlanQuery& q) {
    if (!builtin_rows_match_device(q.dev))
        return std::nullopt;
    int count = 0;
    const TableRow* rows = builtin_plan_table(q.perf_class, count);
    const TableRow* row = rows ? plan_row_for(rows, count, q) : nullptr;
    return row ? std::optional<TableRow>(*row) : std::nullopt;
}

// Calibrated on RTX 5090: tile issue and MMA issue costs, in equivalent bytes.
constexpr std::int64_t kKTileIssueBytes = 8;
constexpr std::int64_t kMmaArmBytesPerInstr = 64;

int resident_of(const GemmRecipe& r, const PlanQuery& q) {
    if (q.dev.smem_per_sm <= 0 || q.dev.regs_per_sm <= 0)
        return 0;
    return std::min(q.dev.smem_per_sm / r.smem, min_ctas_for_ring(r.smem));
}

// TMA overlaps copy and compute; cp.async hides copy latency through residency.
// Resident-scaled waves apply only to two-byte pairs.
std::int64_t cost_of(const GemmRecipe& r, const PlanQuery& q, int resident) {
    const std::int64_t blocks = q.batch * ((std::int64_t)((q.m + r.bm - 1) / r.bm) *
                                           (std::int64_t)((q.n + r.bn - 1) / r.bn));
    if (!q.tma) {
        // Price complete K tiles and divide waves by resident CTA capacity.
        const std::int64_t operand =
            ((q.k + r.kk - 1) / r.kk) * (std::int64_t)r.kk * (r.bm * q.ba + r.bn * q.bb);
        const std::int64_t mu = q.dev.smem_per_sm / r.smem;
        const std::int64_t slots = (std::int64_t)q.dev.sms * mu;
        const std::int64_t waves = slots > 0 ? (blocks + slots - 1) / slots : 1;
        return (operand + (std::int64_t)q.out_elem_bytes * r.bm * r.bn) * waves;
    }
    const bool byte_pair = q.ba == 1 && q.bb == 1;
    const std::int64_t operand = (std::int64_t)q.k * (r.bm * q.ba + r.bn * q.bb);
    const std::int64_t output = (std::int64_t)q.out_elem_bytes * r.bm * r.bn;
    const std::int64_t issue =
        byte_pair ? 0
                  : kKTileIssueBytes * (std::int64_t)r.bm * r.bn * ((q.k + r.kk - 1) / r.kk);
    const std::int64_t mma_arm =
        kMmaArmBytesPerInstr * (std::int64_t)r.bm * r.bn * q.k / (128 * (byte_pair ? 32 : 16));
    const std::int64_t per_cta = std::max(operand + output + issue, mma_arm);
    const std::int64_t slots = (std::int64_t)q.dev.sms * resident;
    const std::int64_t waves = slots > 0 ? (blocks + slots - 1) / slots : 1;
    const std::int64_t w_eff =
        q.ba == 2 && q.bb == 2 ? waves * resident : (blocks + q.dev.sms - 1) / q.dev.sms;
    return per_cta * w_eff;
}

std::optional<PlanDecision> model_plan(const PlanQuery& q) {
    if (q.dev.sms <= 0 || q.m <= 0 || q.n <= 0 || q.k <= 0)
        return std::nullopt;
    std::optional<GemmRecipe> best;
    std::int64_t best_cost = 0;
    with_manifest(q.crosswise > 0, q.ba, q.bb, [&](auto manifest) {
        for_each_recipe<decltype(manifest)>(q.ba, q.bb, [&](GemmRecipe recipe) {
            const int resident = resident_of(recipe, q);
            if (resident <= 0)
                return;
            const std::int64_t cost = cost_of(recipe, q, resident);
            if (!best || cost < best_cost) {
                best = recipe;
                best_cost = cost;
            }
        });
    });
    if (!best)
        return std::nullopt;
    return PlanDecision{*best, plan_raster(q, best->bm, best->bn), "model"};
}

PlanDecision select_plan(const PlanQuery& q) {
    const int mode = gemm_planner_mode();
    if (mode != 2 && !gemm_table_off()) {
        if (auto d = row_plan(q, plan_table_override_source().lookup(q), "override"))
            return *d;
        if (auto d = row_plan(q, plan_table_injected_source().lookup(q), "injected"))
            return *d;
        if (auto d = row_plan(q, builtin_row(q), "builtin"))
            return *d;
    }
    if (mode == 1 || mode == 2)
        if (auto d = model_plan(q))
            return *d;

    // Degraded rows use only M; n=1 passes their exclusive lower bound.
    PlanQuery m_only;
    m_only.m = q.m;
    m_only.n = 1;
    const TableRow* row = plan_row_for(kDegradedPlanRows, 3, m_only);
    if (auto d = row_plan(q, row ? *row : kDegradedPlanRows[0], "degraded"))
        return *d;

    // Preserve the default for empty shapes or missing device facts.
    const TableRow& fallback = kDegradedPlanRows[0];
    return {*recipe_of((int)fallback.cta, fallback.stages, fallback.kk, false, 2, 2),
            0, "degraded"};
}

} // namespace

PlanDecision plan_dispatch(const PlanQuery& q) {
    const PlanDecision decision = select_plan(q);
    log_dispatch(q, decision);
    return decision;
}

} // namespace gemm
} // namespace astrai
