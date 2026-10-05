/* GEMM host planning, config, and recipe vocabulary. */
#include <launcher/plan_types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "plan_table.h"

#include <api/gemm.h>
#include <policy/manifest.cuh>

namespace astrai {
namespace gemm {

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
    if (!recipe || plan_resident_ctas(row->cta, row->stages, row->kk, q) <= 0)
        return std::nullopt;
    return PlanDecision{*recipe,
                        row->raster != 0 ? row->raster : plan_raster(q, recipe->bm, recipe->bn),
                        source};
}

// RTX 5090 fitted mainloop and MMA costs in equivalent bytes, not TMA instruction counts.
constexpr std::int64_t kMainloopBytesPerCellTile = 8;
constexpr std::int64_t kMmaArmBytesPerInstr = 64;

int resident_of(const GemmRecipe& r, const PlanQuery& q) {
    return plan_resident_ctas(static_cast<TileClass>(r.cta), r.stages, r.kk, q);
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
    const std::int64_t k_tiles = (q.k + r.kk - 1) / r.kk;
    const std::int64_t loop_penalty =
        byte_pair ? 0 : kMainloopBytesPerCellTile * (std::int64_t)r.bm * r.bn * k_tiles;
    const std::int64_t mma_instructions =
        k_tiles * (r.kk / q.mma_k) * (r.bm / 16) * (r.bn / 8);
    const std::int64_t mma_arm = kMmaArmBytesPerInstr * mma_instructions;
    const std::int64_t per_cta = std::max(operand + output + loop_penalty, mma_arm);
    const std::int64_t slots = (std::int64_t)q.dev.sms * resident;
    const std::int64_t waves = slots > 0 ? (blocks + slots - 1) / slots : 1;
    const std::int64_t w_eff =
        q.ba == 2 && q.bb == 2 ? waves * resident : (blocks + q.dev.sms - 1) / q.dev.sms;
    return per_cta * w_eff;
}

// geom_cta: rank five work proxies in log space. Smaller is better.
// This is a heuristic ordering, not a prediction of execution time.
double heuristic_cost(const GemmRecipe& named, const PlanQuery& q, int fallback_resident) {
    KernelResources resource{};
    if (q.resources) {
        resource = q.resources(named, q);
    } else {
        // Standalone host callers may have no typed CUDA kernel resolver.
        resource.effective = named;
        resource.resident = fallback_resident;
        if (q.ba + q.bb >= 3 && named.bm == 64 && named.bn == 64 && named.kk == 64) {
            resource.effective.wn = 16;
            resource.effective.threads = 512;
        }
    }
    if (resource.resident <= 0)
        return std::numeric_limits<double>::infinity();
    const auto& r = resource.effective;
    // Integer ceil without n+d-1 overflow; cast before multiplying grid axes.
    auto ceil_div = [](std::int64_t n, int d) { return n / d + (n % d != 0); };
    const double steps = (double)ceil_div(q.k, r.kk);
    const double blocks = (double)q.batch * (double)ceil_div(q.m, r.bm) *
                          (double)ceil_div(q.n, r.bn);
    const double resident = std::min((double)resource.resident,
                                     std::ceil(blocks / q.dev.sms));
    const double waves = std::ceil(blocks / (q.dev.sms * resident));
    const double warps = r.threads / 32.0;
    const double copy = (double)r.kk * (r.bm * q.ba + r.bn * q.bb) / 512.0;
    const double mma = ((double)r.kk / q.mma_k) * (r.bm / 16.0) * (r.bn / 8.0) / warps;
    const double fragments = (double)r.kk * (r.wm * q.ba + r.wn * q.bb) / 512.0;
    const double conversions =
        (double)r.kk * (r.wm * (q.ba < q.bb) + r.wn * (q.bb < q.ba)) / 64.0;
    const double output = (double)q.out_elem_bytes * r.bm * r.bn / 512.0;
    const double issue = q.tma ? 2.0 : copy;

    const double lw = std::log(waves), lr = std::log(resident);
    const double ll = std::log(steps), lp = std::log(warps);
    // log(L*copy + output), without constructing L*copy. The exponential
    // argument is nonpositive; underflow only discards a negligible addend.
    const double a = ll + std::log(copy), b = std::log(output);
    const double log_memory = std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
    const std::array<double, 5> arms = {
        lw + (q.tma ? lr : 0.0) + log_memory,
        lw + lr + ll + lp + std::log(mma),
        lw + ll + std::log(mma + fragments + conversions),
        lw + lr + ll + lp + std::log(fragments + conversions),
        lw + lr + ll + std::log(2.0 * warps + issue),
    };
    // Per-arm normalization is common to all candidates, and the fifth root
    // is monotone: neither changes ranking. Never multiply arms or exp(score).
    double score = 0.0;
    for (double arm : arms)
        score += arm;
    return score;
}

bool same_model_query(const PlanQuery& a, const PlanQuery& b) {
    const auto& x = a.dev;
    const auto& y = b.dev;
    return a.m == b.m && a.n == b.n && a.k == b.k && a.batch == b.batch &&
           a.perf_class == b.perf_class && a.crosswise == b.crosswise &&
           a.ba == b.ba && a.bb == b.bb && a.out_elem_bytes == b.out_elem_bytes &&
           a.mma_k == b.mma_k && a.tma == b.tma && a.resources == b.resources &&
           a.rank3a == b.rank3a && a.rank3b == b.rank3b && a.contiguous == b.contiguous &&
           x.threads_per_sm == y.threads_per_sm && x.ordinal == y.ordinal && x.sms == y.sms && x.smem_max == y.smem_max &&
           x.smem_per_sm == y.smem_per_sm && x.regs_per_sm == y.regs_per_sm &&
           x.l2_bytes == y.l2_bytes && x.cc == y.cc;
}

std::optional<PlanDecision> model_plan(const PlanQuery& q, bool heuristic = false) {
    if (q.dev.sms <= 0 || q.m <= 0 || q.n <= 0 || q.k <= 0 || q.batch <= 0 || q.mma_k <= 0)
        return std::nullopt;
    struct LastModelPlan {
        PlanQuery query{};
        PlanDecision decision{};
        bool heuristic = false;
        bool valid = false;
    };
    static thread_local LastModelPlan last;
    if (last.valid && last.heuristic == heuristic && same_model_query(last.query, q))
        return last.decision;
    std::optional<GemmRecipe> best;
    double best_cost = 0;
    with_manifest(q.crosswise > 0, q.ba, q.bb, [&](auto manifest) {
        for_each_recipe<decltype(manifest)>(q.ba, q.bb, [&](GemmRecipe recipe) {
            const int resident = resident_of(recipe, q);
            if (resident <= 0)
                return;
            const double cost = heuristic ? heuristic_cost(recipe, q, resident)
                                          : (double)cost_of(recipe, q, resident);
            if (!std::isfinite(cost))
                return;
            if (!best || cost < best_cost) {
                best = recipe;
                best_cost = cost;
            }
        });
    });
    if (!best)
        return std::nullopt;
    last = {q,
            {*best, plan_raster(q, best->bm, best->bn), heuristic ? "heuristic" : "model"},
            heuristic, true};
    return last.decision;
}

PlanDecision select_plan(const PlanQuery& q) {
    if (q.m <= 0 || q.n <= 0 || q.k <= 0)
        throw std::invalid_argument("GEMM planner: M, N and K must be greater than zero");
    const int mode = gemm_planner_mode();
    if ((mode == 0 || mode == 1) && !gemm_table_off()) {
        if (auto d = row_plan(q, plan_override_row(q), "override"))
            return *d;
        if (auto d = row_plan(q, plan_injected_row(q), "injected"))
            return *d;
        if (auto d = row_plan(q, plan_builtin_row(q), "builtin"))
            return *d;
    }
    if (mode == 1 || mode == 2 || mode == 3)
        if (auto d = model_plan(q, mode != 2))
            return *d;

    throw std::runtime_error(
        "GEMM planner: no eligible recipe for the selected mode, shape and device");

}

} // namespace

PlanDecision plan_dispatch(const PlanQuery& q) {
    const PlanDecision decision = select_plan(q);
    log_dispatch(q, decision);
    return decision;
}

/* Export instantiated recipes in manifest order for the tuner. */
std::vector<std::vector<int>> tile_vocabulary() {
    const std::pair<int, int> widths[] = {{2, 2}, {2, 1}, {1, 1}};
    std::vector<std::vector<int>> out;
    for (int crosswise = 0; crosswise <= 1; ++crosswise)
        for (const auto& [ba, bb] : widths)
            with_manifest(crosswise != 0, ba, bb, [&](auto manifest) {
                for_each_recipe<decltype(manifest)>(ba, bb, [&](const GemmRecipe& r) {
                    out.push_back({crosswise, ba, bb, r.cta, r.stages, r.kk, r.bm, r.bn,
                                   r.wm, r.wn, r.threads, r.smem});
                });
            });
    return out;
}

/* Names for serialized CTA classes, in enum order. */
std::vector<const char*> tile_class_names() {
    static constexpr const char* kNames[] = {"kSmall64", "kNarrow128x64", "kBig128", "kWide128x256",
                                             "kTall64x128"};
    static_assert((int)TileClass::kTall64x128 == (int)(sizeof(kNames) / sizeof(kNames[0])) - 1,
                  "kNames is indexed by TileClass: keep it in enum order");
    return std::vector<const char*>(kNames, kNames + sizeof(kNames) / sizeof(kNames[0]));
}

} // namespace gemm
} // namespace astrai
