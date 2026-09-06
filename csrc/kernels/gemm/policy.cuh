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
// kernels: the CTA tile and warp tiling arrive as Shape types, the
// cp.async pipeline depth as a stage count.
//
// ElemA / ElemB are independent operand types. The MMA runs on the
// promoted MmaT (gemm_mma_traits): W16A16 passes through, symmetric fp8
// keeps its native mma, and any int8 operand dequantizes in-register to
// bf16 between the fragment load and the mma — kDequantA/kDequantB mark
// those inserts per side (W8A8 inserts both, W8A16 only B).
template <typename ElemA_, typename ElemB_, typename CtaShape_,
          typename WarpShape_, int Stages>
struct GemmTraits {
    using ElemA = ElemA_;
    using ElemB = ElemB_;
    using MmaPair = gemm_mma_traits<ElemA_, ElemB_>;
    using MmaT = typename MmaPair::MmaT;
    using ElemTraitsA = gemm_elem_traits<ElemA_>;
    using ElemTraitsB = gemm_elem_traits<ElemB_>;

    using CtaShape = CtaShape_;
    using WarpShape = WarpShape_;
    static constexpr int kBlockM = CtaShape_::kM;
    static constexpr int kBlockN = CtaShape_::kN;
    static constexpr int kK = CtaShape_::kK;
    static constexpr int kStages = Stages;
    static constexpr int kWarpM = WarpShape_::kM;
    static constexpr int kWarpN = WarpShape_::kN;

    static constexpr int kElemBytesA = ElemTraitsA::kBytes;
    static constexpr int kElemBytesB = ElemTraitsB::kBytes;
    // MMA shape follows the promoted compute type; dequantized fragments
    // are brought to it in-register (dequant.cuh).
    static constexpr int kMmaK = gemm_elem_traits<MmaT>::kMmaK;
    static constexpr bool kDequantA = MmaPair::kDequantA;
    static constexpr bool kDequantB = MmaPair::kDequantB;

    // Derived geometry: warp tiles tile the CTA. The smem budget is
    // layout-aware, so it lives in GemmSmem (below).
    static constexpr int kWarpsM = kBlockM / kWarpM;
    static constexpr int kWarpsN = kBlockN / kWarpN;
    static constexpr int kCtaThreads = kWarpsM * kWarpsN * 32;
    static_assert(kWarpsM * kWarpM == kBlockM && kWarpsN * kWarpN == kBlockN,
                  "warp tiles must exactly tile the CTA");
    static_assert(kWarpM % 16 == 0 && kWarpN % 8 == 0,
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
    static constexpr int kBytes =
        kRingDepth * Traits::kBlockM * Traits::kK * Traits::kElemBytesA +
        kRingDepth * Traits::kBlockN * Traits::kK * Traits::kElemBytesB;
    static constexpr int kMinCtas = kBytes <= 48 * 1024 ? 2 : 1;
};

// Tile recipe (CUTLASS-style configuration type): one named bundle of CTA
// shape, warp tiling, pipeline depth and loop specialization. A policy
// composes a tile config with operand dtypes and layout tags; the host
// planner (gemm.cuh) enumerates the manifest below — extending the launch
// ladder with a new geometry means adding one alias here and one planner
// branch, never re-spelling positional ints.
template <typename CtaShape_, typename WarpShape_, int Stages_, bool FastLoop_>
struct GemmTileConfig {
    using CtaShape = CtaShape_;
    using WarpShape = WarpShape_;
    static constexpr int kStages = Stages_;
    static constexpr bool kFastLoop = FastLoop_;
};

// Production tile manifest — the tuned configs launch_plan dispatches to
// (L20-measured; see the crossover tables in cuda_kernels.md). Big CTA:
// 128x128 of 8 warps x 64x32, kK=64, 2-stage full ring; the fast
// (predication-free) loop exists only for dual-congruous staging. Narrow:
// 128x64, the wave-filling and fat-output route. Small CTA: 64x64 of 4
// warps x 32x32 — the 24KB s2 variant keeps 4 CTAs/SM resident, the 32KB
// s3 variant trades that for a deeper pipeline on multi-wave grids.
using TileBig128x128 = GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 2, false>;
using TileBigFast = GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 2, true>;
using TileNarrow128x64 = GemmTileConfig<Shape<128, 64, 64>, Shape<32, 32>, 2, true>;
using TileSmall64s2 = GemmTileConfig<Shape<64, 64, 64>, Shape<32, 32>, 2, true>;
using TileSmall64s3 = GemmTileConfig<Shape<64, 64, 64>, Shape<32, 32>, 3, true>;

template <typename ElemA_, typename ElemB_, typename LayoutA_, typename LayoutB_,
          typename Tile_, typename LayoutOut_ = RowMajor,
          typename OutT_ = __nv_bfloat16, bool StreamOut_ = false>
struct GemmPolicy {
    using Tile = Tile_;
    using Traits = GemmTraits<ElemA_, ElemB_, typename Tile_::CtaShape,
                              typename Tile_::WarpShape, Tile_::kStages>;
    using LayoutTagA = LayoutA_;
    using LayoutTagB = LayoutB_;
    // Output orientation (CUTLASS LayoutC): direction lives in the type,
    // the row stride lives in GemmParams::out_ld.
    using LayoutTagOut = LayoutOut_;
    // Output element type: bf16 (default, the fused-linear convention) or
    // fp32 (accumulated outputs, e.g. training dX/dW). The epilogue stages
    // and copies out through OutElem<OutT> packing facts.
    using OutT = OutT_;
    static constexpr bool kStreamOut = StreamOut_;
    static constexpr bool kFastLoop = Tile_::kFastLoop;
    using Smem = GemmSmem<Traits, LayoutA_, LayoutB_>;
    // Flattened for __launch_bounds__, which takes no dependent type names.
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr int kMinCtas = Smem::kMinCtas;
    static constexpr int kSmemBytes = Smem::kBytes;
};

// fp8 convenience aliases: format-parameterized names over the generic
// policy (A and B share the fp8 type), kept for the binding's FP8Format
// dispatch and the C tests.
template <FP8Format Fmt, typename CtaShape, typename WarpShape, int Stages>
using Fp8GemmTraits =
    GemmTraits<fp8_elem_t<Fmt>, fp8_elem_t<Fmt>, CtaShape, WarpShape, Stages>;

template <FP8Format Fmt_, typename LayoutA_, typename LayoutB_, typename Tile_,
          typename LayoutOut_ = RowMajor, typename OutT_ = __nv_bfloat16,
          bool StreamOut_ = false>
using Fp8GemmPolicy =
    GemmPolicy<fp8_elem_t<Fmt_>, fp8_elem_t<Fmt_>, LayoutA_, LayoutB_, Tile_,
               LayoutOut_, OutT_, StreamOut_>;

}  // namespace gemm
}  // namespace astrai
