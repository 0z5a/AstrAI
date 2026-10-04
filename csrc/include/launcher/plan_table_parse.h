#pragma once
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <launcher/plan_row.h>

namespace astrai {
namespace gemm {

/*
 * Row file format: one row per line, whitespace-separated
 *   m_min m_max n_min n_max perf_class crosswise cta stages raster
 *   [k [k_min k_max [min_ctas_per_sm [min_wave_permille]]]]
 * '#' starts a comment; perf_class 0..3 or -1; crosswise 0..2 or -1; cta is the
 * TileClass ordinal (the policy/manifest.cuh enum order); stages 2..5 parse, but no
 * ladder instantiates a tile past s3 — plan_from_row rejects a deeper ring
 * outright, so a stale sweep row naming s4/s5 falls to the next source. The
 * trailing 'k' defaults to kTableRowK, which is what the sweep
 * scripts leave off.
 * The trailing forms are additive — a row that omits them behaves exactly as
 * it did before they existed, which is what keeps hand-edited tuning files and
 * older sweeps valid. Invalid lines are warn-and-skip: a malformed row must
 * never block a launch the fallback would serve.
 */

// lo <= v <= hi, for the fields whose legal values are a contiguous interval.
inline constexpr bool in_range(int v, int lo, int hi) { return v >= lo && v <= hi; }

/*
 * A band is (min, max]: min exclusive, 0 = unbounded, so the sentinel stays
 * out of the ordering test. max == min is an empty band — legal, and simply
 * never matched.
 */
inline constexpr bool row_band_ok(int64_t min, int64_t max) { return max == 0 || max >= min; }

/*
 * First out-of-range field, or nullptr: one check per line so the warning
 * names the hand-edited column that is wrong.
 */
inline const char* plan_row_error(const TableRow& row, int fields) {
    if (fields != kRowFields && fields != kRowFieldsLegacyK && fields != kRowFieldsKband &&
        fields != kRowFieldsWave && fields != kRowFieldsWavePermille)
        return "field count";
    if (row.m_min < 0 || row.n_min < 0)
        return "band min < 0";
    if (!row_k_supported(row.kk))
        return "k (want 32 or 64)";
    if (!row_band_ok(row.m_min, row.m_max))
        return "m band (max < min)";
    if (!row_band_ok(row.n_min, row.n_max))
        return "n band (max < min)";
    if (!row_band_ok(row.k_min, row.k_max))
        return "k band (max < min)";
    if (row.k_min < 0 || row.k_max < 0)
        return "k band min < 0";
    if (row.min_ctas_per_sm < 0)
        return "min_ctas_per_sm < 0";
    if (row.min_wave_permille < 0)
        return "min_wave_permille < 0";
    if (!in_range(row.perf_class, -1, kMaxPerfClass))
        return "perf_class";
    if (!in_range(row.crosswise, -1, 2))
        return "crosswise (-1..2)";
    if (!row_stages_supported(row.stages))
        return "stages (want 2..5)";
    return nullptr;
}

// Warn-and-skip: a hand-edit typo costs one row, not the table.
inline void warn_bad_row(const std::string& path, int lineno, const char* why) {
    std::fprintf(stderr, "[gemm-plan-table] %s:%d: ignoring row: bad %s\n", path.c_str(), lineno,
                 why);
}

/*
 * One row-file line (label names the source in warnings). Mutated in place
 * (the '#' comment cut); both the file and runtime-injection readers go
 * through here so the two cannot drift.
 */
inline void
parse_plan_table_line(char* line, const char* label, int lineno, std::vector<TableRow>& rows) {
    if (char* hash = std::strchr(line, '#'); hash != nullptr)
        *hash = '\0';
    long long m_min, m_max, n_min, n_max;
    int perf_class, crosswise, cta, stages, raster;
    /*
     * sscanf leaves a variable alone when its conversion fails, so a row
     * that omits the trailing fields keeps the defaults here: the legacy
     * and k-less field counts need no repair pass.
     */
    int kk = kTableRowK;
    long long k_min = 0, k_max = 0;
    int min_ctas_per_sm = 0;
    int min_wave_permille = 0;
    const int got =
        std::sscanf(line, " %lld %lld %lld %lld %d %d %d %d %d %d %lld %lld %d %d", &m_min, &m_max,
                    &n_min, &n_max, &perf_class, &crosswise, &cta, &stages, &raster, &kk, &k_min,
                    &k_max, &min_ctas_per_sm, &min_wave_permille);
    if (got == EOF)
        return; // blank or comment-only line
    /*
     * cta is read as the TileClass ordinal, so it is the one field checked
     * before there is a row to validate; plan_row_error takes the rest.
     */
    if (!in_range(cta, 0, kTileClassCount - 1)) {
        warn_bad_row(label, lineno, "cta index");
        return;
    }
    const TableRow row{static_cast<TileClass>(cta),
                       m_min,
                       m_max,
                       n_min,
                       n_max,
                       perf_class,
                       crosswise,
                       stages,
                       raster,
                       kk,
                       k_min,
                       k_max,
                       min_ctas_per_sm,
                       min_wave_permille};
    if (const char* bad = plan_row_error(row, got); bad != nullptr) {
        warn_bad_row(label, lineno, bad);
        return;
    }
    rows.push_back(row);
}

inline bool parse_plan_table_file(const std::string& path, std::vector<TableRow>& rows) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr)
        return false;
    char line[256];
    int lineno = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        ++lineno;
        parse_plan_table_line(line, path.c_str(), lineno, rows);
    }
    std::fclose(f);
    return true;
}

/*
 * The same parser over in-memory row text: the runtime channel and the file
 * path accept identical syntax. Returns the surviving row count.
 */
inline int
parse_plan_table_text(const std::string& text, const char* label, std::vector<TableRow>& rows) {
    const int before = (int)rows.size();
    std::string line;
    int lineno = 0;
    for (std::size_t pos = 0; pos <= text.size(); ++pos) {
        const char c = pos < text.size() ? text[pos] : '\n';
        if (c != '\n' && c != '\r') {
            line.push_back(c);
            continue;
        }
        ++lineno;
        line.push_back('\0');
        parse_plan_table_line(&line[0], label, lineno, rows);
        line.clear();
        /*
         * The trailing newline of the final chunk loops once more with an
         * empty string; an empty line parses to nothing, so the extra pass
         * is harmless.
         */
    }
    return (int)rows.size() - before;
}


} // namespace gemm
} // namespace astrai
