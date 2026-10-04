#pragma once
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <launcher/plan_row.h>
#include <launcher/plan_table_builtin.h>
#include <launcher/plan_table_parse.h>

namespace astrai {
namespace gemm {

/*
 * One row tier's backing store: mutex-guarded, replaced wholesale (never
 * mutated in place), lookups copy the row out under the lock — a concurrent
 * install cannot dangle a pointer a launched plan holds. Two instances:
 * override (the configure rows channel, outranks everything) and injected
 * (the autotuner's measured winners; ranked below the env file, which is
 * the experimenter's). `source` is the install spec, kept so the config
 * state can re-install exactly (re-emitting parsed rows would lose the
 * gates the text format cannot express).
 */
class RowSource {
  public:
    void set_from(std::string source, std::vector<TableRow> rows) {
        std::lock_guard<std::mutex> g(mutex_);
        rows_ = std::move(rows);
        source_ = std::move(source);
    }
    void clear() { set_from({}, {}); }
    std::string source() const {
        std::lock_guard<std::mutex> g(mutex_);
        return source_;
    }
    std::optional<TableRow> lookup(const PlanQuery& q) const {
        std::lock_guard<std::mutex> g(mutex_);
        const TableRow* row = plan_row_for(rows_.data(), (int)rows_.size(), q);
        if (row == nullptr)
            return std::nullopt;
        return *row;
    }
    size_t size() const {
        std::lock_guard<std::mutex> g(mutex_);
        return rows_.size();
    }

  private:
    mutable std::mutex mutex_;
    std::vector<TableRow> rows_;
    std::string source_;
};

inline RowSource& plan_table_override_source() {
    static RowSource source;
    return source;
}

inline RowSource& plan_table_injected_source() {
    static RowSource source;
    return source;
}

/*
 * The planner-rank vocabulary, one place: the strings configure() takes
 * and config_state() returns for GemmConfig::planner.
 */
inline constexpr const char* kPlannerModeNames[] = {"table", "hybrid", "model"};
inline constexpr int kPlannerModeCount = 3;
inline bool parse_planner_mode(const std::string& name, int& out) {
    for (int i = 0; i < kPlannerModeCount; ++i)
        if (name == kPlannerModeNames[i]) {
            out = i;
            return true;
        }
    return false;
}

/*
 * Legacy ASTR_GEMM_* env vars, read once on first touch; configure() writes
 * the atomics directly and bypasses this.
 */
inline void gemm_config_seed_once() {
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

/*
 * Resolved views (unset falls to the default, never to a later env read);
 * the three launch-side knobs' resolved views are plan_types.h's.
 */
inline int gemm_planner_mode() {
    gemm_config_seed_once();
    const int v = gemm_config().planner.load(std::memory_order_relaxed);
    return v < 0 ? 1 : v; // default: hybrid (model fills what no row owns)
}
inline bool gemm_table_off() {
    gemm_config_seed_once();
    return gemm_config().table_off.load(std::memory_order_relaxed) > 0;
}


} // namespace gemm
} // namespace astrai
