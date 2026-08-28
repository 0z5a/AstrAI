#pragma once
// Kernel policy layer: shared-memory budget, occupancy hint and the
// single Policy type the kernel and collectives take (CUTLASS-style
// consolidation of traits + layout tags + scheduling knobs).

#include <type_traits>

#include "../common.h"

namespace astrai {
namespace fp8 {

// m16n8k32 (see astrai::mma_shape<fp8 type>::k in common/mma.cuh)
constexpr int kMmaK = 32;

// Layout-aware shared-memory budget and occupancy hint. Every operand ring
// holds kStages+1 buffers: the load for tile i+kStages targets slot
// (i-1)%(kStages+1) — already consumed — so neither load path needs a
// post-compute barrier (one __syncthreads per k-tile; see the design notes
// in docs/developer/cuda_kernels.md). The 48KB static watermark picks the
// resident-CTA hint for __launch_bounds__.
template <typename Traits, typename LayoutA, typename LayoutB>
struct Fp8GemmSmem {
    // Crosswise (direct-load) operands: A ColMajor storage, B RowMajor
    // storage (B's tag is relative to the canonical [K][N]).
    static constexpr bool kDirectA = std::is_same_v<LayoutA, ColMajor>;
    static constexpr bool kDirectB = std::is_same_v<LayoutB, RowMajor>;
    static constexpr int kRingDepth = Traits::kStages + 1;
    static constexpr int kBytes =
        kRingDepth * (Traits::kBlockM + Traits::kBlockN) * Traits::kK;
    static constexpr int kMinCtas = kBytes <= 48 * 1024 ? 2 : 1;
};

template <FP8Format Fmt_, int BlockM_, int BlockN_, typename LayoutA_,
          typename LayoutB_, int WarpM_, int WarpN_, int kK_, int Stages_,
          int GroupRaster_, bool StreamOut_ = false, bool FastLoop_ = false>
struct Fp8GemmPolicy {
    using Traits =
        Fp8GemmTraits<Fmt_, BlockM_, BlockN_, kK_, Stages_, WarpM_, WarpN_>;
    using LayoutTagA = LayoutA_;
    using LayoutTagB = LayoutB_;
    static constexpr int kGroupRaster = GroupRaster_;
    static constexpr bool kStreamOut = StreamOut_;
    static constexpr bool kFastLoop = FastLoop_;
    using Smem = Fp8GemmSmem<Traits, LayoutA_, LayoutB_>;
    // Flattened for __launch_bounds__, which takes no dependent type names.
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr int kMinCtas = Smem::kMinCtas;
    static constexpr int kSmemBytes = Smem::kBytes;
};

}  // namespace fp8
}  // namespace astrai
