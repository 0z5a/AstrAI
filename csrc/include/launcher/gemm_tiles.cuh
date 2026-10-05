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

/* Resolve each manifest tile to one policy. TMA can reject a descriptor and fall back. */
template <bool UseTma, typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut, typename OutT, typename Schedule>
struct TileLauncher {
    const GemmParams& p;
    cudaStream_t stream;

    template <typename Tile> bool run() const {
        using Widened = warp_widened_t<ElemA, ElemB, Tile>;
        using TileT = reclaim_fallback_t<Widened, ElemA, ElemB, OutT>;
        using Policy = GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, TileT, LayoutOut, OutT,
                                  false, UseTma, Schedule::kMx, false, Schedule>;
        if constexpr (UseTma)
            return launch_policy_tma<Policy>(p, stream);
        else {
            launch_policy<Policy>(p, stream);
            return true;
        }
    }
};

/*
 * Plan -> Policy: dispatch_tile picks the manifest entry, the launcher
 * applies the ladder's substitutions; stages >= 3 selects the deep-ring
 * sibling. Params by value — the raster decision lands in the copy the
 * kernel receives. The schedule selects the MMA cell.
 */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename LayoutOut = RowMajor,
          typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync>
void launch_plan(GemmParams p, const PlanDecision& d, cudaStream_t stream) {
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
    if constexpr (Schedule::kTma && kCongruous && sizeof(ElemA) <= 2 && sizeof(ElemB) <= 2) {
        if (!gemm_tma_staging_disabled() && astrai::device_facts().cc >= 90 &&
            dispatch_tile<manifest_for<ElemA, ElemB, RowMajor, ColMajor>>(
                d, TileLauncher<true, ElemA, ElemB, RowMajor, ColMajor, LayoutOut, OutT, Schedule>{p, stream}))
            return;
    }
    dispatch_tile<manifest_for<ElemA, ElemB, LayoutA, LayoutB>>(
        d, TileLauncher<false, ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT, Schedule>{p, stream});
}

} // namespace gemm
} // namespace astrai
