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

// Row tiers copy lookup results under a mutex; installs replace the whole table.
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
