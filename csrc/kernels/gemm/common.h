#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

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

    const float* __restrict__ scale = nullptr;

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
};

}  // namespace gemm
}  // namespace astrai
