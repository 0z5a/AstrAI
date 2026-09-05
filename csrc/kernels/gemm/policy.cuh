#pragma once
// Kernel policy layer: shared-memory budget, occupancy hint and the
// single Policy type the kernel and collectives take (CUTLASS-style
// consolidation of traits + layout tags + scheduling knobs). Dtype-generic:
// parameterized on the operand element type; the per-dtype facts come from
// gemm_elem_traits (gemm/common.h). The fp8 names below are thin aliases
// over FP8Format for the binding's format dispatch.

#include <cuda_fp8.h>
#include <type_traits>

#include "gemm/common.h"
#include "quantize/common.h"

namespace astrai {
namespace gemm {

using quant::FP8Format;

// Operand element type for one fp8 format (fp8 convenience alias layer).
template <FP8Format Fmt>
using fp8_elem_t =
    std::conditional_t<Fmt == FP8Format::E5M2, __nv_fp8_e5m2, __nv_fp8_e4m3>;

// Compile-time tile configuration, mirroring KernelTraits in the attention
// kernels: CTA tile, warp tile (WarpM x WarpN — e.g. 64x32 on the 128x128
// CTA, 32x32 on the 64x64 small CTA) and cp.async pipeline depth.
template <typename ElemT_, int BlockM, int BlockN, int K, int Stages,
          int WarpM = 64, int WarpN = 32>
struct GemmTraits {
    using ElemT = ElemT_;
    using ElemTraits = gemm_elem_traits<ElemT_>;

    static constexpr int kBlockM = BlockM;
    static constexpr int kBlockN = BlockN;
    static constexpr int kK = K;
    static constexpr int kStages = Stages;
    static constexpr int kWarpM = WarpM;
    static constexpr int kWarpN = WarpN;

    static constexpr int kElemBytes = ElemTraits::kBytes;
    static constexpr int kMmaK = ElemTraits::kMmaK;
    static constexpr bool kNeedsDequant = ElemTraits::kNeedsDequant;

    // Derived geometry: warp tiles tile the CTA. The smem budget is
    // layout-aware, so it lives in GemmSmem (below).
    static constexpr int kWarpsM = BlockM / WarpM;
    static constexpr int kWarpsN = BlockN / WarpN;
    static constexpr int kCtaThreads = kWarpsM * kWarpsN * 32;
    static_assert(kWarpsM * WarpM == BlockM && kWarpsN * WarpN == BlockN,
                  "warp tiles must exactly tile the CTA");
    static_assert(WarpM % 16 == 0 && WarpN % 8 == 0,
                  "warp tile must be a multiple of the m16n8 MMA shape");
};

// Layout-aware shared-memory budget and occupancy hint. Every operand ring
// holds kStages+1 buffers: the load for tile i+kStages targets slot
// (i-1)%(kStages+1) — already consumed — so neither load path needs a
// post-compute barrier (one __syncthreads per k-tile; see the design notes
// in docs/developer/cuda_kernels.md). The 48KB static watermark picks the
// resident-CTA hint for __launch_bounds__.
template <typename Traits, typename LayoutA, typename LayoutB>
struct GemmSmem {
    // Crosswise (direct-load) operands: A ColMajor storage, B RowMajor
    // storage (B's tag is relative to the canonical [K][N]).
    static constexpr bool kDirectA = std::is_same_v<LayoutA, ColMajor>;
    static constexpr bool kDirectB = std::is_same_v<LayoutB, RowMajor>;
    static constexpr int kRingDepth = Traits::kStages + 1;
    static constexpr int kBytes = kRingDepth * (Traits::kBlockM + Traits::kBlockN) *
                                  Traits::kK * Traits::kElemBytes;
    static constexpr int kMinCtas = kBytes <= 48 * 1024 ? 2 : 1;
};

template <typename ElemT_, int BlockM_, int BlockN_, typename LayoutA_,
          typename LayoutB_, int WarpM_, int WarpN_, int kK_, int Stages_,
          int GroupRaster_, bool StreamOut_ = false, bool FastLoop_ = false>
struct GemmPolicy {
    using Traits = GemmTraits<ElemT_, BlockM_, BlockN_, kK_, Stages_, WarpM_, WarpN_>;
    using LayoutTagA = LayoutA_;
    using LayoutTagB = LayoutB_;
    static constexpr int kGroupRaster = GroupRaster_;
    static constexpr bool kStreamOut = StreamOut_;
    static constexpr bool kFastLoop = FastLoop_;
    using Smem = GemmSmem<Traits, LayoutA_, LayoutB_>;
    // Flattened for __launch_bounds__, which takes no dependent type names.
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr int kMinCtas = Smem::kMinCtas;
    static constexpr int kSmemBytes = Smem::kBytes;
};

// fp8 convenience aliases: format-parameterized names over the generic
// policy, kept for the binding's FP8Format dispatch and the C tests.
template <FP8Format Fmt, int BlockM, int BlockN, int K, int Stages,
          int WarpM = 64, int WarpN = 32>
using Fp8GemmTraits = GemmTraits<fp8_elem_t<Fmt>, BlockM, BlockN, K, Stages, WarpM, WarpN>;

template <FP8Format Fmt_, int BlockM_, int BlockN_, typename LayoutA_,
          typename LayoutB_, int WarpM_, int WarpN_, int kK_, int Stages_,
          int GroupRaster_, bool StreamOut_ = false, bool FastLoop_ = false>
using Fp8GemmPolicy =
    GemmPolicy<fp8_elem_t<Fmt_>, BlockM_, BlockN_, LayoutA_, LayoutB_, WarpM_,
               WarpN_, kK_, Stages_, GroupRaster_, StreamOut_, FastLoop_>;

}  // namespace gemm
}  // namespace astrai
