#pragma once
/* Manifest tile selection and TMA/cp.async policy resolution. */
#include <tuple>
#include <type_traits>

#include <launcher/gemm_launch.cuh>

namespace astrai {
namespace gemm {

/*
 * Manifest dispatch (CUTLASS builder-table style): (CTA class, ring depth,
 * k-tile depth) selects one manifest entry; the resolver maps it to a Policy
 * and launches.
 */
template <typename Manifest, typename Resolver>
bool dispatch_tile(const PlanDecision& d, const Resolver& resolve) {
    return std::apply(
        [&d, &resolve](auto... tiles) {
            return (... || (tile_class<decltype(tiles)>() == static_cast<TileClass>(d.recipe.cta) &&
                            decltype(tiles)::kStages == d.recipe.stages &&
                            (int)decltype(tiles)::CtaShape::kK == d.recipe.kk &&
                            resolve.template run<decltype(tiles)>()));
        },
        Manifest{});
}

/*
 * Narrow twin of a big tile for the reclaim fallback, same ring depth as
 * the tile it replaces (the planner priced that ring). No kK=32 narrow s3
 * exists — that big tile falls to its s2 twin.
 */
template <typename Tile>
using narrow_fallback_t = std::conditional_t<
    Tile::CtaShape::kK == 32,
    Tile_128x64x32_W32x32_S2,
    std::conditional_t<(Tile::kStages >= 3), Tile_128x64x64_W32x32_S3, Tile_128x64x64_W32x32_S2>>;

/*
 * The reclaim chain every instantiation terminates in: output outgrows the
 * ring -> narrow twin; the kK=32 narrow (18KB ring) still cannot hold 4B/elem
 * -> small CTA (24KB reclaims every output <= 4B/elem). Identity whenever
 * the tile fits, so the launchers apply it unconditionally; the dispatch
 * names every manifest tile as a potential substitute, so the chain is
 * load-bearing even for unplanned geometries.
 */
template <typename Tile, typename ElemA, typename ElemB, typename OutT>
using reclaim_fallback_t = std::conditional_t<
    reclaim_fits<Tile, ElemA, ElemB, OutT>(),
    Tile,
    std::conditional_t<reclaim_fits<narrow_fallback_t<Tile>, ElemA, ElemB, OutT>(),
                       narrow_fallback_t<Tile>,
                       Tile_64x64x64_W16x32_S2>>;

/*
 * TMA ladder resolver: the launch_plan gate already guarantees
 * dual-congruous 1-/2-byte operands, so only a reclaim overflow swaps the
 * CTA. conditional_t keeps every alias instantiable (an if-constexpr branch
 * still NAMES its dead types — the kernel's static_assert requires it).
 */
template <typename ElemA, typename ElemB, typename LayoutOut, typename OutT, bool UseMx = false>
struct TmaLauncher {
    const GemmParams& p;
    cudaStream_t stream;
    template <typename Tile> bool run() const {
        /*
         * The reclaim chain is applied unconditionally (identity whenever the
         * tile fits): the dispatch walks name every manifest tile, so even
         * the narrow/small classes need an out when an instantiation's output
         * outgrows their rings (the kK=32 narrow vs a 4B/elem output).
         */
        using Widened = warp_widened_t<ElemA, ElemB, Tile>;
        using TileT = reclaim_fallback_t<Widened, ElemA, ElemB, OutT>;
        return launch_policy_tma<GemmPolicy<ElemA, ElemB, RowMajor, ColMajor, TileT, LayoutOut,
                                            OutT, false, true, UseMx>>(p, stream);
    }
};

/*
 * cp.async ladder resolver: one substitution — a fat output (fp32) that
 * cannot reclaim the ring routes to the narrow CTA (same math, lower
 * reuse). Narrow and small pass through. (A second substitution — the big
 * CTA downgraded to a predicated-loop twin on crosswise staging — measured
 * 9-19% SLOWER in interleaved A/B, 2026-09-16; see policy.cuh's
 * GemmTileConfig note. Buried.)
 */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename LayoutOut,
          typename OutT,
          bool UseMx = false>
struct CpAsyncLauncher {
    GemmParams p;
    cudaStream_t stream;
    template <typename Tile> bool run() const {
        /*
         * Warp widening + reclaim chain, both unconditional (identity when
         * the tile fits); the rule lives in policy.cuh's warp_widened_t.
         */
        using Widened = warp_widened_t<ElemA, ElemB, Tile>;
        using TileT = reclaim_fallback_t<Widened, ElemA, ElemB, OutT>;
        launch_policy<GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, TileT, LayoutOut, OutT, false,
                                 false, UseMx>>(p, stream);
        return true;
    }
};

/*
 * Plan -> Policy: dispatch_tile picks the manifest entry, the launcher
 * applies the ladder's substitutions; stages >= 3 selects the deep-ring
 * sibling. Params by value — the raster decision lands in the copy the
 * kernel receives. UseMx threads the block_scale cell through.
 */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename LayoutOut = RowMajor,
          typename OutT = __nv_bfloat16,
          bool UseMx = false>
void launch_plan_impl(GemmParams p, const PlanDecision& d, cudaStream_t stream) {
    p.raster = d.raster;
    /*
     * Dual-congruous (crosswise 0): the only layout pair TMA can describe —
     * both operands staged as-is, so the descriptors are encodable.
     */
    constexpr bool kCongruous = crosswise_of<LayoutA, LayoutB>() == 0;
    /*
     * TMA staging first when the layout pair and dtypes allow it (the
     * planner's stage/tile decisions are shared): sm_90+ device, no
     * kill switch, and every descriptor encodable — else the cp.async
     * twin below runs unchanged.
     */
    if constexpr (kCongruous && sizeof(ElemA) <= 2 && sizeof(ElemB) <= 2) {
        if (!gemm_tma_staging_disabled() && astrai::device_facts().cc >= 90 &&
            dispatch_tile<manifest_for<ElemA, ElemB, RowMajor, ColMajor>>(
                d, TmaLauncher<ElemA, ElemB, LayoutOut, OutT, UseMx>{p, stream}))
            return;
    }
    dispatch_tile<manifest_for<ElemA, ElemB, LayoutA, LayoutB>>(
        d, CpAsyncLauncher<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT, UseMx>{p, stream});
}

/*
 * Planner entry: symmetric fp8 rides the sm_120 block_scale cell unless
 * knocked out.
 */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename LayoutOut = RowMajor,
          typename OutT = __nv_bfloat16>
void launch_plan(GemmParams p, const PlanDecision& d, cudaStream_t stream) {
    constexpr bool kMxCell =
        (std::is_same_v<ElemA, __nv_fp8_e4m3> && std::is_same_v<ElemB, __nv_fp8_e4m3>) ||
        (std::is_same_v<ElemA, __nv_fp8_e5m2> && std::is_same_v<ElemB, __nv_fp8_e5m2>);
    if constexpr (kMxCell) {
        /*
         * cc is CC-tens (120 = CC 12.0). One half of a contract: CMake
         * emits the sm_120a slice exactly when "120" is in the arch list,
         * so the route fires only where that image exists.
         */
        if (!gemm_mx_cell_disabled() && astrai::device_facts().cc == 120) {
            launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT, true>(p, d, stream);
            return;
        }
    }
    launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT>(p, d, stream);
}


} // namespace gemm
} // namespace astrai
