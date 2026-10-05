#pragma once
#include <algorithm>
#include <cstdint>

#include <launcher/plan_types.h>
#include <policy.cuh>
#include <utils/device.cuh>

namespace astrai {
namespace gemm {


/*
 * Ring K when the row file omits the field (the manifest holds kK twins;
 * the measured winner is kK=32 on most shapes but not all).
 */
static constexpr int kTableRowK = 64;
/*
 * Field counts the parser reads: current, legacy 9 (no k), +k-band 12,
 * +wave-gate 13, +wave-permille 14.
 */
static constexpr int kRowFields = 10;
static constexpr int kRowFieldsLegacyK = 9;
static constexpr int kRowFieldsKband = 12;
static constexpr int kRowFieldsWave = 13;
static constexpr int kRowFieldsWavePermille = 14;

/*
 * Upper bound of the perf_class field, mirroring GemmPerfClass's last
 * enumerator (kF8A8, launcher/plan_types.h — where the enum lives).
 */
static constexpr int kMaxPerfClass = 3;

/*
 * The k-tile depths the manifests carry; any other depth matches no tile
 * and launches nothing, so it is rejected.
 */
inline constexpr bool row_k_supported(int kk) { return kk == 32 || kk == 64 || kk == 128; }

/*
 * Ring depths a row may name. Enumerated, not a range, so a gap cannot pass
 * as "inside 2..5": s4/s5 stay parseable for old sweep files, but no ladder
 * instantiates past s3 (the s4/s5 twins were a measured wash, removed).
 */
inline constexpr bool row_stages_supported(int stages) {
    return stages == 2 || stages == 3 || stages == 4 || stages == 5;
}
/*
 * One tuned row. Bands are (min, max] — min exclusive, 0 unbounded — K band
 * included. perf_class is the GemmPerfClass id, crosswise the operand count
 * (-1 = any for both); raster 0 resolves through plan_raster at this row's
 * geometry; kk defaults to kTableRowK. The K band exists because the recipe
 * flips with K (wide CTA wants K > ~256, kK=32 twin wins short K hardest).
 *
 * The wave gates match only while this tile's grid covers that much of the
 * machine: bare form grid >= n * sms, wave form grid * 1000 >= n * sms *
 * resident (resident priced from the row's ring). 0 = no gate. Prefer the
 * wave form — dimensionless, survives a different SM count or smem; literal
 * bounds (a measured crossover, "a row above answers this band") are
 * device-calibrated and want a re-measure elsewhere.
 */
struct TableRow {
    TileClass cta;
    int64_t m_min;
    int64_t m_max;
    int64_t n_min;
    int64_t n_max;
    int perf_class;
    int crosswise;
    int stages;
    int raster;
    int kk = kTableRowK;
    int64_t k_min = 0;
    int64_t k_max = 0;
    int min_ctas_per_sm = 0;
    int min_wave_permille = 0;
};

/*
 * CTA geometry of one class, off kTileClassCta (self-asserted against the
 * tiles), so the numbers cannot drift from what the ladders instantiate.
 */
inline constexpr void plan_row_geometry(TileClass cta, int& bm, int& bn) {
    bm = kTileClassCta[(int)cta][0];
    bn = kTileClassCta[(int)cta][1];
}

/*
 * Everything a plan decision is priced against is PlanQuery — in launcher/plan_types.h
 * (the kernel-side vocabulary), so this header compiles against types the
 * launchers already see.
 */

/*
 * CTAs of one plan's tile per SM, or 0 when unpriced/unlaunchable. Minimum
 * of smem-per-SM over the ring and the __launch_bounds__ hint — the hint is
 * a FLOOR on the real count (the exact figure needs a driver query a pure
 * planner cannot make), so resident_model <= resident_true fires the wave
 * gate early, never late. Exact for the 512-thread tiles where the register
 * file binds (64 regs x 512 x 2 = 64K) — the only gated ring today.
 */
inline int plan_resident_ctas(TileClass cta, int stages, int kk, const PlanQuery& q) {
    const DeviceFacts& dev = q.dev;
    if (dev.smem_per_sm <= 0 || dev.regs_per_sm <= 0)
        return 0;
    int bm = 0, bn = 0;
    plan_row_geometry(cta, bm, bn);
    const int ring = ring_smem_bytes(bm, bn, kk, stages, q.ba, q.bb);
    if (ring > dev.smem_max)
        return 0;
    return std::min(dev.smem_per_sm / ring, min_ctas_for_ring(ring));
}

/*
 * First matching row wins (generated tables are non-overlapping). q.k <= 0
 * is the open-K reading (degraded rows, no-depth callers); q.dev.sms <= 0
 * skips gated rows rather than guessing, and the lookup still ends at the
 * degraded rows — planning stays total.
 */
inline const TableRow* plan_row_for(const TableRow* rows, int count, const PlanQuery& q) {
    for (int i = 0; i < count; ++i) {
        const TableRow& r = rows[i];
        if (q.m <= r.m_min)
            continue;
        if (r.m_max != 0 && q.m > r.m_max)
            continue;
        if (q.n <= r.n_min)
            continue;
        if (r.n_max != 0 && q.n > r.n_max)
            continue;
        if (r.perf_class != -1 && r.perf_class != q.perf_class)
            continue;
        if (r.crosswise != -1 && r.crosswise != q.crosswise)
            continue;
        /*
         * An open row (both bounds 0) matches any K, including the k <= 0
         * callers; a bounded row only matches a real depth.
         */
        if (r.k_min != 0 || r.k_max != 0) {
            if (q.k <= 0)
                continue;
            if (q.k <= r.k_min)
                continue;
            if (r.k_max != 0 && q.k > r.k_max)
                continue;
        }
        /*
         * Wave gates: the row's own geometry prices the grid, so a row cannot
         * state a fill it could not itself satisfy. min_ctas_per_sm is the
         * bare CTAs-per-SM form; min_wave_permille the wave form, whose
         * resident term is priced from the row's ring on this device.
         */
        if (r.min_ctas_per_sm > 0 || r.min_wave_permille > 0) {
            if (q.dev.sms <= 0)
                continue;
            int bm, bn;
            plan_row_geometry(r.cta, bm, bn);
            const int64_t grid = ((q.m + bm - 1) / bm) * ((q.n + bn - 1) / bn) * q.batch;
            if (r.min_ctas_per_sm > 0 && grid < (int64_t)r.min_ctas_per_sm * q.dev.sms)
                continue;
            if (r.min_wave_permille > 0) {
                const int resident = plan_resident_ctas(r.cta, r.stages, r.kk, q);
                if (resident <= 0)
                    continue;
                if (grid * 1000 < (int64_t)r.min_wave_permille * q.dev.sms * resident)
                    continue;
            }
        }
        return &r;
    }
    return nullptr;
}


} // namespace gemm
} // namespace astrai
