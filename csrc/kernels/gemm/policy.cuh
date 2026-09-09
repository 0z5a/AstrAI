#pragma once
// Kernel policy layer: shared-memory budget, occupancy hint and the
// single Policy type the kernel and collectives take (CUTLASS-style
// consolidation of traits + layout tags + scheduling knobs). Dtype-generic:
// parameterized on the operand element type; the per-dtype facts come from
// gemm_elem_traits (gemm/common.h).

#include <cuda_fp8.h>
#include <tuple>
#include <type_traits>

#include "common/mma.cuh"
#include "common/tensor.cuh"
#include "gemm/common.h"
#include "quantize/common.h"

namespace astrai {
namespace gemm {

// Compile-time tile configuration, mirroring KernelTraits in the attention
// kernels: the CTA tile and warp tiling arrive as Shape types, the
// cp.async pipeline depth as a stage count.
//
// ElemA / ElemB are independent operand types. The MMA runs on the
// promoted MmaT (gemm_mma_traits): W16A16 passes through, symmetric fp8
// and symmetric int8 keep their native mma (fp32 / int32 accumulators),
// and a lone int8 or fp8 operand against bf16 dequantizes in-register
// between the fragment load and the mma — kDequantA/kDequantB mark those
// inserts per side (W8A16 only B).
//
// UseMx swaps the symmetric-fp8 cell for the sm_120 block_scale cell
// (MxMmaOp); other pairs ignore it.
template <typename ElemA_, typename ElemB_, typename CtaShape_,
          typename WarpShape_, int Stages, bool UseMx = false>
struct GemmTraits {
    using ElemA = ElemA_;
    using ElemB = ElemB_;
    using MmaPair = gemm_mma_traits<ElemA_, ElemB_>;
    using MmaT = typename MmaPair::MmaT;
    // The exact mma cell <MmaT, MmaT, shape> (common/mma.cuh): one
    // type carries the instruction's K extent, register counts and the
    // accumulator type (fp32 for the float families, s32 for the s8 pair).
    static constexpr bool kMxCell =
        UseMx && (std::is_same_v<MmaT, __nv_fp8_e4m3> ||
                  std::is_same_v<MmaT, __nv_fp8_e5m2>);
    using MmaOp = std::conditional_t<
        kMxCell, astrai::MxMmaOp<MmaT>,
        astrai::MmaOp<MmaT, MmaT, typename astrai::MmaShapeFor<MmaT>::type>>;
    using AccT = typename MmaOp::AccT;
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
    // MMA shape follows the promoted compute type (the shape trait above is
    // the single source); dequantized fragments are brought to it
    // in-register (dequant.cuh).
    static constexpr int kMmaK = astrai::MmaShapeFor<MmaT>::type::kK;
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
    // Warp accumulator geometry, one definition for mainloop and epilogue:
    // m16n8 mma cells on the (kMt, kNt) warp tile grid.
    static constexpr int kMt = kWarpM / 16;
    static constexpr int kNt = kWarpN / 8;
    using AccTensor =
        Tensor<ArrayEngine<typename MmaOp::CFrag, kMt * kNt>,
               CellLayout<kNt>>;
};

// Ring-budget formula, one source for GemmSmem, launch_plan's epilogue
// reclaim check and the host planner's recipe feasibility gate (gemm.cuh):
// every operand ring holds kStages+1 buffers of k * (bm*ba + bn*bb) bytes.
constexpr int ring_smem_bytes(int bm, int bn, int k, int stages,
                              int ba, int bb) {
    return (stages + 1) * k * (bm * ba + bn * bb);
}

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
        ring_smem_bytes(Traits::kBlockM, Traits::kBlockN, Traits::kK,
                        Traits::kStages, Traits::kElemBytesA,
                        Traits::kElemBytesB);
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
// warps x 32x32 — the 24KB s2 variant keeps 4 CTAs/SM resident, the 32KB s3
// variant trades that for a deeper pipeline on multi-wave grids.
//
// The s3 deep-ring variants (humming's _fit_num_stages rule) spend the
// smem headroom a thinner operand pair leaves under the 96KB budget the
// fat bf16 pair already fills: 2B x 1B reaches s3 on the big CTA
// (4 buffers x 24KB), 2B x 2B on the narrow CTA — one extra in-flight
// k-tile of DRAM latency to hide on long-K shapes.
using TileBig128x128 = GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 2, false>;
using TileBigFast = GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 2, true>;
using TileNarrow128x64 = GemmTileConfig<Shape<128, 64, 64>, Shape<32, 32>, 2, true>;
// 8-warp small CTA (16x32 warp tiles, 256 threads/CTA — 2026-09-09 A/B):
// same 64x64x64 footprint and staging, double the per-SM thread count for
// the underfed short-M shapes; mainloop traits adapt with this alias.
using TileSmall64s2 = GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 2, true>;
using TileSmall64s3 = GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 3, true>;
using TileBig128x128s3 = GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 3, false>;
using TileBigFastS3 = GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 3, true>;
using TileNarrow128x64s3 = GemmTileConfig<Shape<128, 64, 64>, Shape<32, 32>, 3, true>;

// CTA class of a tile config, derived from its CTA geometry — the dispatch
// key the launch ladders select on (GemmPlan::Cta in gemm.cuh is this enum).
enum class TileClass { kSmall64, kNarrow128x64, kBig128 };

template <typename Tile>
constexpr TileClass tile_class() {
    if constexpr (Tile::CtaShape::kM == 128 && Tile::CtaShape::kN == 128)
        return TileClass::kBig128;
    else if constexpr (Tile::CtaShape::kM == 128 && Tile::CtaShape::kN == 64)
        return TileClass::kNarrow128x64;
    else
        return TileClass::kSmall64;
}

// The dispatch manifest (CUTLASS builder-table style): every tuned recipe
// the launch ladders in gemm.cuh select over. The ladders index this list
// by the plan's CTA class and depth bit — extending them is one alias here
// plus one planner branch, never a re-spelled per-site ladder. The big
// entries carry the fast variant; the cp.async ladder downgrades to the
// non-fast twin for crosswise staging at its resolver.
using TileManifest = std::tuple<
    TileBigFast, TileBigFastS3,
    TileNarrow128x64, TileNarrow128x64s3,
    TileSmall64s2, TileSmall64s3>;

template <typename ElemA_, typename ElemB_, typename LayoutA_, typename LayoutB_,
          typename Tile_, typename LayoutOut_ = RowMajor,
          typename OutT_ = __nv_bfloat16, bool StreamOut_ = false,
          bool UseTma_ = false, bool UseMxMma_ = false,
          bool StoreWriteThrough_ = false>
struct GemmPolicy {
    using Tile = Tile_;
    using Traits = GemmTraits<ElemA_, ElemB_, typename Tile_::CtaShape,
                              typename Tile_::WarpShape, Tile_::kStages,
                              UseMxMma_>;
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
    // Output streaming store (__stwt, write-through, bypasses the L2
    // write-back stage): the fused-linear output is read-once and never
    // reused, so keeping it out of L2 reserves the cache for the reused
    // weights/activations. Separate from kStreamOut (__stcs, evict-first);
    // kStoreWriteThrough wins when both are set.
    static constexpr bool kStoreWriteThrough = StoreWriteThrough_;
    // TMA staging (sm_90+): congruous-only by construction — the launcher
    // instantiates these policies solely for dual-congruous layout pairs
    // with aligned operands; staging layouts and fragment addressing are
    // identical, only the load/wait discipline changes (tma.cuh).
    static constexpr bool kUseTma = UseTma_;
    static_assert(!UseTma_ || (sizeof(ElemA_) <= 2 && sizeof(ElemB_) <= 2),
                  "TMA staging covers the 1-/2-byte congruous dtypes");
    static constexpr bool kFastLoop = Tile_::kFastLoop;
    using Smem = GemmSmem<Traits, LayoutA_, LayoutB_>;
    // TMA budgets the 1024B ring-base alignment pad plus the full/empty
    // mbarrier pair per ring slot (tma.cuh); the residency hint stays
    // ring-based.
    static constexpr int kTmaExtra =
        UseTma_ ? 1024 + 2 * (Tile_::kStages + 1) * 8 : 0;
    // Flattened for __launch_bounds__, which takes no dependent type names.
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr int kMinCtas = Smem::kMinCtas;
    static constexpr int kSmemBytes = Smem::kBytes + kTmaExtra;
};

}  // namespace gemm
}  // namespace astrai
