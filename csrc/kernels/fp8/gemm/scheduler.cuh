#pragma once
// Tile scheduler: the linear CTA id maps to (block_m, block_n) in grouped
// (L2-friendly) raster — consecutive CTAs share one B column stripe — or
// plain N-fastest raster (kRasterGroup=0, the measured best for dX's
// crosswise-B layouts where grouping was neutral).

namespace astrai {
namespace fp8 {

template <int kRasterGroup>
struct Fp8GemmTileScheduler {
    static __device__ int2 tile(const uint3& block, const dim3& blocks) {
        if constexpr (kRasterGroup > 0) {
            constexpr int kGroupM = kRasterGroup;
            const int bid = int(block.y) * int(blocks.x) + int(block.x);
            const int group_first_m = (bid / (kGroupM * int(blocks.x))) * kGroupM;
            const int group_rows =
                min(int(blocks.y) - group_first_m, kGroupM);  // M-tail group is short
            return int2{group_first_m + bid % group_rows,
                        (bid % (kGroupM * int(blocks.x))) / group_rows};
        } else {
            return int2{int(block.y), int(block.x)};
        }
    }
};

}  // namespace fp8
}  // namespace astrai
