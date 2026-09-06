#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// GEMM-family pure POD/traits header — dtype-neutral: layout tags, element
// traits and the unified parameter POD shared by every element-type
// specialization (fp8 today, future bf16/half policies); raw __nv_* type
// spellings only.

namespace astrai {
namespace gemm {

// Operand storage tags (CUTLASS-style) relative to the canonical matrices
// A [M][K] / B [K][N]: A RowMajor = [M][K] (default), A ColMajor = [K][M],
// B RowMajor = [K][N], B ColMajor = [N][K] (the nn.Linear weight). Selection
// is by type at compile time (see gemm.cuh's stage loads).
struct RowMajor {};
struct ColMajor {};

// Element-type traits: the per-dtype facts the policy/smem/load layers
// derive geometry from. Adding a dtype = adding a specialization here plus
// an mma_shape<InT> in common/mma.cuh (fragment layout + mma.sync shape);
// everything downstream consumes only these constants.
//
//   kBytes        sizeof one operand element (smem budget scaling)
//   kMmaK         K extent of one tensor-core MMA instruction
//   kNeedsDequant epilogue applies the combined dequant scale
template <typename T>
struct gemm_elem_traits;

template <>
struct gemm_elem_traits<__nv_fp8_e4m3> {
    static constexpr int kBytes = 1;
    static constexpr int kMmaK = 32;  // mma.sync.m16n8k32 (sm_89+)
    static constexpr bool kNeedsDequant = true;
};

template <>
struct gemm_elem_traits<__nv_fp8_e5m2> {
    static constexpr int kBytes = 1;
    static constexpr int kMmaK = 32;  // mma.sync.m16n8k32 (sm_89+)
    static constexpr bool kNeedsDequant = true;
};

template <>
struct gemm_elem_traits<__nv_bfloat16> {
    static constexpr int kBytes = 2;
    static constexpr int kMmaK = 16;  // mma.sync.m16n8k16 (sm_80+)
    static constexpr bool kNeedsDequant = false;
};

// int8 quantized operand (weight-only W8A16 or dynamic W8A8): staged
// packed, dequantized in-register to the MMA compute type between the
// fragment load and the mma.sync (dequant.cuh), so its kMmaK never feeds
// the tile geometry — that comes from the promoted pair traits below.
template <>
struct gemm_elem_traits<int8_t> {
    static constexpr int kBytes = 1;
    static constexpr int kMmaK = 16;  // via the bf16 promotion (unused directly)
    static constexpr bool kNeedsDequant = true;
};

// MMA compute type for one operand pair — the dtype promotion every
// quantized-GEMM family (humming-style) routes through: the tensor-core
// input type both operands are brought to before mma.sync. Symmetric fp8
// pairs run their native fp8 mma; any pair involving int8 (W8A16, W8A8,
// A8W16) promotes to bf16 m16n8k16 with in-register dequant of the int8
// side(s). Symmetric bf16 (W16A16) passes through untouched. kDequantA/B
// are per-operand: W8A8 dequantizes both sides, W8A16 only B.
template <typename ElemA, typename ElemB>
struct gemm_mma_traits {
    using MmaT = std::conditional_t<std::is_same_v<ElemA, int8_t> ||
                                        std::is_same_v<ElemB, int8_t>,
                                    __nv_bfloat16, ElemA>;
    static constexpr bool kDequantA = !std::is_same_v<ElemA, MmaT>;
    static constexpr bool kDequantB = !std::is_same_v<ElemB, MmaT>;
};

// Unified GEMM parameter POD, mirroring AttentionParams: one struct flows
// through the kernels; each kernel touches only the fields it needs.
struct GemmParams {
    // Operands + output; scale is the combined dequant step (device
    // scalar; fp8 policies apply it in the epilogue, bf16 policies ignore
    // it). Optional bf16 bias fuses into the epilogue (fp32 add before the
    // single bf16 rounding); null disables.
    const void* __restrict__ a_ptr = nullptr;
    const void* __restrict__ b_ptr = nullptr;
    const void* __restrict__ bias_ptr = nullptr;
    void* __restrict__ out_ptr = nullptr;

    // Per-operand dequant scales, folded multiplicatively into the
    // epilogue (the mma accumulates the raw quantized product):
    //   a_scale — length a_scale_m: 0 = device scalar applied to the whole
    //             output (per-tensor activation dequant), m = per-row [m];
    //   b_scale — length b_scale_n: 0 = device scalar, n = per-output-
    //             channel [n] (weight-only quantization);
    // null / 0 disable each side.  Grouped-along-K scales cannot fold here
    // and belong in the mainloop (not implemented).
    const float* __restrict__ a_scale = nullptr;
    const float* __restrict__ b_scale = nullptr;
    int a_scale_m = 0;
    int b_scale_n = 0;

    // Batched (bmm) geometry: grid.z steps these element strides (0
    // broadcasts the operand across batches).
    int batch = 1;
    // Single extents and row strides fit int for LLM shapes; kernels
    // promote to int64. Batch strides are extent *products* (k*m, k*n,
    // m*n) and can cross the int32 boundary on large bmms.
    int m, n, k;
    // Physical leading dims (row strides in elements) of A, B and the
    // output; out_ld lets non-contiguous outputs (slices of a larger
    // buffer) cost nothing — the epilogue writes out[row * out_ld + col].
    // The output orientation is not data: it rides the policy's LayoutOut
    // tag (see policy.cuh).
    int a_ld, b_ld, out_ld;
    int64_t a_batch_stride = 0;
    int64_t b_batch_stride = 0;
    int64_t out_batch_stride = 0;


    // Raster order (runtime knob — the plan layer picks it from the
    // problem's aspect; see plan_gemm):
    //   >0 — grouped raster, group of `raster` M-tile rows, M walked
    //        fastest inside a group (consecutive CTAs share one B column
    //        stripe); best when there are at least as many M tiles as N;
    //   <0 — mirrored: group of -raster N-tile columns, N walked fastest
    //        (consecutive CTAs share one A row stripe) — the wide-output
    //        shapes (e.g. dW = g^T @ x);
    //    0 — plain N-fastest raster (no grouping).
    int raster = 8;
};

}  // namespace gemm
}  // namespace astrai
