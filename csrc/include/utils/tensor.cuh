/*
 * Tensor<Engine, Layout> separates storage from indexing. PtrEngine and
 * ArrayEngine provide pointer and register storage; ComposedLayout, RingLayout,
 * and CellLayout map staged chunks, ring slots, and accumulator cells. make_ring
 * constructs a ring and stage_of slices a slot. All operations inline away.
 */

#pragma once

#include <cstdint>

#include <utils/define.cuh>
#include <utils/swizzle.cuh>

namespace astrai {

// Engines

/*
 * Shared/global-memory storage: the engine knows the element, the layout
 * knows the address map.
 */
template <typename T> struct PtrEngine {
    using Elem = T;
    T* ptr;
    DEVICE_FORCEINLINE T* base() const { return ptr; }
};

/*
 * Register-array storage (cute's Array role — an mma fragment cell IS an
 * array). Passed BY REFERENCE to the mma/ldmatrix emitters so the
 * registers stay in place — no address arithmetic can appear at the seams.
 */
template <typename T, int N> struct ArrayEngine {
    using Elem = T;
    T storage[N];
    DEVICE_FORCEINLINE T& operator[](int i) { return storage[i]; }
    DEVICE_FORCEINLINE const T& operator[](int i) const { return storage[i]; }
    DEVICE_FORCEINLINE T* base() { return storage; }
    DEVICE_FORCEINLINE const T* base() const { return storage; }
};

// Layouts: address maps; chunk-grid operations are dtype-blind

/*
 * Ring layout: slot rotation over a per-stage chunk grid — the staged
 * ring expressed as one (slot, row, chunk) map instead of a bespoke ring
 * object. Chunk units are 16B, so the byte budget is dtype-independent.
 */
template <typename StageLay_, int kSlots_> struct RingLayout {
    using Stage = StageLay_;
    static constexpr bool kChunkUnit = true;
    static constexpr int kSlots = kSlots_;
    static constexpr int kStageChunks = StageLay_::kRows * StageLay_::kChunks;
    static constexpr int kStageBytes = kStageChunks * 16;
    static constexpr int kTotalBytes = kSlots * kStageBytes;
    DEVICE_FORCEINLINE uint32_t operator()(uint32_t slot, uint32_t row, uint32_t chunk) const {
        return slot * (uint32_t)kStageChunks + StageLay_{}(row, chunk);
    }
};

/*
 * Cell layout: an element-unit row-major (m, n) grid — the accumulator's
 * (mt, nt) mma-cell coordinates.
 */
template <int kCols> struct CellLayout {
    static constexpr bool kChunkUnit = false;
    DEVICE_FORCEINLINE uint32_t operator()(uint32_t m, uint32_t n) const {
        return m * (uint32_t)kCols + n;
    }
};

// Tensor

/* Chunk layouts scale 16B indices by Elem; CellLayout addresses whole engine cells. */
template <typename EngineT, typename LayoutT> struct Tensor {
    using Elem = typename EngineT::Elem;
    using Layout = LayoutT;
    static constexpr int kChunkElems = LayoutT::kChunkUnit ? 16 / (int)sizeof(Elem) : 1;

    EngineT engine;
    LayoutT layout;

    /*
     * Keep row and swizzled-chunk offsets separate in 32-bit arithmetic, then
     * widen once for the pointer add. The XOR depends on row alone; linearizing
     * it behind row*stride regressed W8A8 by up to 29% (see GEMM notes).
     */
    template <bool kChunk = LayoutT::kChunkUnit, std::enable_if_t<kChunk, int> = 0>
    DEVICE_FORCEINLINE Elem* operator()(int row, int col) const {
        constexpr int kShift = log2_const<kChunkElems>::value;
        const uint32_t off = (uint32_t)row * (uint32_t)(LayoutT::kChunks * kChunkElems) +
                             (layout.chunk_of((uint32_t)row, (uint32_t)(col >> kShift)) << kShift) +
                             (uint32_t)(col & (kChunkElems - 1));
        return engine.base() + (ptrdiff_t)off;
    }
    // Chunk-unit 3-coordinate ring view: layout(slot, row, chunk).
    template <bool kChunk = LayoutT::kChunkUnit, std::enable_if_t<kChunk, int> = 0>
    DEVICE_FORCEINLINE Elem* operator()(int slot, int row, int col) const {
        constexpr int kShift = log2_const<kChunkElems>::value;
        const typename LayoutT::Stage stage{};
        const uint32_t off = (uint32_t)slot * (uint32_t)(LayoutT::kStageChunks * kChunkElems) +
                             (uint32_t)row * (uint32_t)(LayoutT::Stage::kChunks * kChunkElems) +
                             (stage.chunk_of((uint32_t)row, (uint32_t)(col >> kShift)) << kShift) +
                             (uint32_t)(col & (kChunkElems - 1));
        return engine.base() + (ptrdiff_t)off;
    }
    /*
     * Element-unit cell view: layout(m, n) -> &engine cell. Non-const: the
     * accumulator rides non-const references through the mainloop/epilogue.
     */
    template <bool kChunk = LayoutT::kChunkUnit, std::enable_if_t<!kChunk, int> = 0>
    DEVICE_FORCEINLINE Elem* operator()(int m, int n) {
        return engine.base() + (size_t)layout((uint32_t)m, (uint32_t)n);
    }
};

// Tensor factories and slicing

// Construct the staged ring tensor over a raw shared-memory carve.
template <typename ElemT, typename StageLay, int kSlots>
DEVICE_FORCEINLINE Tensor<PtrEngine<ElemT>, RingLayout<StageLay, kSlots>> make_ring(char* smem) {
    return {PtrEngine<ElemT>{reinterpret_cast<ElemT*>(smem)}, {}};
}

/*
 * Slice one slot's tile out of the ring (slot = tile % kSlots) — cute's
 * tensor slicing; .engine.ptr also serves the layout-agnostic writers
 * (cp.async / TMA boxes stage through the raw address). The smem carve
 * points and the TMA barrier placement measure against RingLayout's byte
 * facts.
 */
template <typename ElemT, typename StageLay, int kSlots>
DEVICE_FORCEINLINE Tensor<PtrEngine<ElemT>, StageLay>
stage_of(const Tensor<PtrEngine<ElemT>, RingLayout<StageLay, kSlots>>& ring, int64_t tile) {
    return {PtrEngine<ElemT>{ring.engine.ptr + (size_t)(tile % kSlots) *
                                                   RingLayout<StageLay, kSlots>::kStageChunks *
                                                   (16 / (int)sizeof(ElemT))},
            {}};
}

} // namespace astrai
