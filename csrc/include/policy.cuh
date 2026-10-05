#pragma once
/* One GEMM kernel policy composed from compute traits and a tile recipe. */
#include <policy/manifest.cuh>

namespace astrai {
namespace gemm {

template <typename ElemA_,
          typename ElemB_,
          typename LayoutA_,
          typename LayoutB_,
          typename Tile_,
          typename LayoutOut_ = RowMajor,
          typename OutT_ = __nv_bfloat16,
          bool StreamOut_ = false,
          bool UseTma_ = false,
          bool UseMxMma_ = false,
          bool StoreWriteThrough_ = false,
          typename Schedule_ = MmaSync>
struct GemmPolicy {
    using Schedule = Schedule_;
    using Tile = Tile_;
    using Traits = GemmTraits<ElemA_,
                              ElemB_,
                              typename Tile_::CtaShape,
                              typename Tile_::WarpShape,
                              Tile_::kStages,
                              UseMxMma_>;
    using LayoutTagA = LayoutA_;
    using LayoutTagB = LayoutB_;
    /*
     * Output orientation (CUTLASS LayoutC): direction in the type, stride
     * in GemmParams::out_ld. OutT: bf16 (fused-linear convention) or fp32
     * (accumulated outputs, e.g. training dX/dW).
     */
    using LayoutTagOut = LayoutOut_;
    using OutT = OutT_;
    static constexpr bool kStreamOut = StreamOut_;
    /*
     * __stwt write-through store: the fused-linear output is read-once, so
     * keeping it out of L2 reserves the cache for reused weights/activations.
     * Separate from kStreamOut (__stcs, evict-first); write-through wins
     * when both are set.
     */
    static constexpr bool kStoreWriteThrough = StoreWriteThrough_;
    /*
     * TMA staging (sm_90+): congruous-only by construction — these policies
     * are instantiated solely for dual-congruous layout pairs with aligned
     * operands; staging layouts and fragment addressing are identical, only
     * the load/wait discipline changes (tma.cuh).
     */
    static constexpr bool kUseTma = UseTma_;
    static_assert(!UseTma_ || (sizeof(ElemA_) <= 2 && sizeof(ElemB_) <= 2),
                  "TMA staging covers the 1-/2-byte congruous dtypes");
    using Smem = GemmSmem<Traits, LayoutA_, LayoutB_>;
    /*
     * TMA budgets the 1024B ring-base alignment pad plus the full/empty
     * mbarrier pair per ring slot (tma.cuh); the residency hint stays
     * ring-based.
     */
    static constexpr int kTmaExtra = UseTma_ ? 1024 + 2 * (Tile_::kStages + 1) * 8 : 0;
    // Flattened for __launch_bounds__, which takes no dependent type names.
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr int kMinCtas = Smem::kMinCtas;
    static constexpr int kSmemBytes = Smem::kBytes + kTmaExtra;
};


} // namespace gemm
} // namespace astrai
