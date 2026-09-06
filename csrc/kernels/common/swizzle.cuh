// Unified staging-swizzle vocabulary, CUTLASS-style (cute): a swizzle is a
// TYPE composed with a layout into the staged tile's address map —
//
//   using SmemLayoutA = decltype(
//       composition(Swizzle<Bits, Shift>{},
//                   Layout<Shape<Rows, Chunks>, Stride<Chunks, 1>>{}));
//
// where the functor is a bijection over the LINEAR 16B-chunk index,
//
//   Swizzle<Bits, Shift>{}(L)  =  L ^ ((L >> Shift) & (2^Bits - 1))
//
// (cute measures the offset in elements; 16B chunks keep one (Bits, Shift)
// pair dtype-independent), and the layout the row-major map — the one
// fixed point of cute's layout algebra every staging site here is. One
// declared instance per staged tile names the whole map: the stage
// loaders, the fragment readers and the folded lane-offset mirrors all
// consume the same type. Every staged tile in the kernel families is one
// (Bits, Shift) pair (the XOR stays inside the low chunk field, so the
// row bits never carry):
//
//   congruous operand tile, 2B elems: <3, 3>  — the TMA SWIZZLE_128B pattern
//   congruous operand tile, 1B elems: <2, 3>  — the TMA SWIZZLE_64B pattern
//   crosswise trans tile (16-bit):    <3, log2 chunks> — custom
//   epilogue wide-row tile:           <log2 chunks, log2 chunks> — custom
//
// The Shift == 3 members ARE the hardware TMA swizzle modes (Bits 1/2/3 =
// 32/64/128B spans) — kTmaMode marks them, the gate for any future TMA
// staging; the descriptor's swizzle enum derives from the same Bits.
// (TMA applies the XOR to the ABSOLUTE shared-memory address, so a TMA
// consumer either aligns the tile to 1024B or phases the index by the
// tile's own address bits.) Every application folds to one IMAD plus one
// XOR immediate; the vocabulary is zero-cost.
//
// The Shape/Stride extent vocabulary this layer's layouts are written in
// lives in shape.cuh (shared with the policy and mma trait layers); only
// the swizzle-specific carriers are defined here.

#pragma once

#include <cstdint>

#include "shape.cuh"

namespace astrai {

template <int Bits, int Shift>
struct Swizzle {
    static_assert(Bits >= 0 && Shift >= 0 && Bits + Shift <= 16,
                  "swizzle field out of the 16B-chunk index range");
    static constexpr int kBits = Bits;
    static constexpr int kShift = Shift;
    static constexpr uint32_t kMask = (uint32_t(1) << Bits) - 1;
    // The Shift==3 members are exactly the TMA swizzle modes.
    static constexpr bool kTmaMode = Shift == 3 && Bits >= 1 && Bits <= 3;
    __device__ __forceinline__ uint32_t operator()(uint32_t linear) const {
        return linear ^ ((linear >> Shift) & kMask);
    }
};

// Layout carriers: the Shape above (see shape.cuh) carries the extents,
// Stride the affine
// map. All staged tiles are row-major 16B-chunk grids, so the row stride
// is the chunk count and the column stride one.
template <int... Ns>
struct Stride;

template <typename ShapeT, typename StrideT>
struct Layout;

template <int Rows, int Chunks, int RowStride, int ColStride>
struct Layout<Shape<Rows, Chunks>, Stride<RowStride, ColStride>> {
    static_assert(RowStride == Chunks && ColStride == 1,
                  "staged chunk grids are row-major packed");
    static constexpr int kRows = Rows;
    static constexpr int kChunks = Chunks;
    __device__ __forceinline__ uint32_t operator()(uint32_t row,
                                                   uint32_t chunk) const {
        return row * Chunks + chunk;
    }
};

// composition(Swizzle, Layout) — cute's composed-layout idiom: the swizzle
// bijection applied to the layout's offset. operator() is the composition
// pre-folded to its closed two-coordinate form, chunk' = (chunk & ~kMask) |
// ((chunk ^ (row >> kRowShift)) & kMask): for every chunk < kChunks the
// swizzle's XOR value ((L >> Shift) & kMask — row bits) is narrower than
// the chunk field, so it never spills into the row term — identical map to
// Swz{}(LayT{}(row, chunk)), but the XOR derives from the row ALONE and
// computes in parallel with the chunk extraction instead of serializing
// behind the row*stride IMAD (CUTLASS 2.x's iterators apply the swizzle
// the same way). Tensors dispatch to this op (common/tensor.cuh); nothing
// re-derives strides at call sites.
template <typename SwzT, typename LayT>
struct ComposedLayout {
    using Swz = SwzT;
    using Lay = LayT;
    // 16B-chunk domain flag for the tensor layer (common/tensor.cuh):
    // the (Bits, Shift) pair stays dtype-blind; the tensor scales.
    static constexpr bool kChunkUnit = true;
    static constexpr int kRows = LayT::kRows;
    static constexpr int kChunks = LayT::kChunks;
    static_assert(kChunks >= 1 && (kChunks & (kChunks - 1)) == 0,
                  "the XOR swizzle needs a power-of-two chunk count");
    static_assert(SwzT::kBits <= log2_const<kChunks>::value,
                  "closed two-coordinate form needs the XOR inside the "
                  "chunk field");
    static constexpr int kRowShift = SwzT::kShift - log2_const<kChunks>::value;
    static_assert(kRowShift >= 0,
                  "swizzle source must start inside the row field");
    static constexpr uint32_t kMask = SwzT::kMask;
    // The layout's chunk-map op: the swizzled chunk coordinate alone (the
    // row term stays out — the tensor scales the two terms separately in
    // 32-bit so the address chain never widens to 64-bit).
    __device__ __forceinline__ uint32_t chunk_of(uint32_t row,
                                                 uint32_t chunk) const {
        const uint32_t swz = (row >> kRowShift) & kMask;
        return (chunk & ~kMask) | ((chunk ^ swz) & kMask);
    }
    // Linear chunk index: row-major layout over the swizzled chunk.
    __device__ __forceinline__ uint32_t operator()(uint32_t row,
                                                   uint32_t chunk) const {
        return row * (uint32_t)kChunks + chunk_of(row, chunk);
    }
};

template <typename SwzT, typename LayT>
constexpr ComposedLayout<SwzT, LayT> composition(SwzT, LayT) {
    return {};
}

}  // namespace astrai
