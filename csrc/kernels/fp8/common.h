#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

// Pure POD/traits header — no .cuh/CUDA-kernel includes; raw __nv_* type
// spellings only.

namespace astrai {
namespace fp8 {

// Compile-time FP8 format: E4M3 (forward / high precision, max 448) or
// E5M2 (gradient / large dynamic range, max 57344).
enum class FP8Format : int {
    E4M3 = 0,
    E5M2 = 1,
};

// Operand memory layouts as types (CUTLASS-style tags). The tag names the
// storage order of the raw buffer relative to the operand's canonical GEMM
// matrix — A is [M][K], B is [K][N]:
//   A RowMajor = [M][K] storage (K-contiguous rows; the default)
//   A ColMajor = [K][M] storage (M-contiguous; A^T)
//   B RowMajor = [K][N] storage (N-contiguous; the plain a @ b operand)
//   B ColMajor = [N][K] storage (K-contiguous; the nn.Linear weight layout)
// Empty tags: selection happens by type at compile time (see load_operand_tile).
struct RowMajor {};
struct ColMajor {};

// Transpose of a layout tag: the same buffer with the rows and contract dims
// swapped. B's tag is relative to the canonical [K][N] GEMM matrix, so the
// stage-load (which views any operand as [rows][contract]) sees the transposed
// tag — this trait makes that inversion explicit.
template <typename Layout>
struct transpose_layout;
template <>
struct transpose_layout<RowMajor> {
    using type = ColMajor;
};
template <>
struct transpose_layout<ColMajor> {
    using type = RowMajor;
};
template <typename Layout>
using transpose_layout_t = typename transpose_layout<Layout>::type;

// Compile-time tile configuration, mirroring KernelTraits<HEAD_DIM, BC,
// WARPS, STAGES> in the attention kernels. `Fmt` selects the FP8 conversion
// and the MMA PTX mnemonic; the remaining parameters shape the CTA tile and
// the cp.async pipeline depth.
template <FP8Format Fmt, int BlockM, int BlockN, int K, int Stages>
struct Fp8GemmTraits {
    static constexpr FP8Format kFormat = Fmt;
    static constexpr int kBlockM = BlockM;
    static constexpr int kBlockN = BlockN;
    static constexpr int kK = K;
    static constexpr int kStages = Stages;
    static constexpr bool kIsE5M2 = (Fmt == FP8Format::E5M2);
    static constexpr __nv_fp8_interpretation_t kNvFormat =
        kIsE5M2 ? __NV_E5M2 : __NV_E4M3;
    static constexpr float kFp8Max = kIsE5M2 ? 57344.0f : 448.0f;
};

// Quantize-kernel parameter POD: BF16 -> FP8 with fused amax and optional
// delayed-scaling ring finalization. Separate from FP8Params so each
// operator owns exactly the fields it touches (the GEMM never reads amax /
// ring state). Same NSDMI rationale: amax / ring_state gate optional paths
// via null checks. Still an aggregate, still trivially copyable.
struct FP8QuantizeParams {
    // BF16 input and FP8 output buffers; scale_a is the quantization step
    // (device scalar). amax_a (may be null) is zero-initialized by the
    // binding and receives the raw-domain absolute maximum.
    const void* __restrict__ a_ptr = nullptr;
    void* __restrict__ out_ptr = nullptr;
    const float* __restrict__ scale_a = nullptr;
    float* __restrict__ amax_a = nullptr;

    // Optional delayed-scaling ring finalization. ring_state packs
    // [hist[ring_len] | scale | counter] with ring_len = numel - 2. When
    // non-null and amax_a is set, the last-finishing block records the
    // measured amax into hist[ring_idx], reduces the window and publishes
    // the next step's scale (max(hist) / fp8_max / 2^ring_margin) — the
    // fused replacement for the eager hist-write / max / scale-write chain,
    // at zero extra launches. The counter slot is a persistent zero-armed
    // int32 (float bits) electing the last block each launch.
    float* ring_state = nullptr;
    int ring_len = 0;
    int ring_idx = 0;
    int ring_margin = 0;

    // Element count (only the elementwise quantize kernel uses it).
    int total = 0;
};

// Unified GEMM parameter POD, mirroring AttentionParams: one struct flows
// through the pre-quantized GEMM kernels. Each kernel touches only the
// fields it needs; buffers are raw pointers packed by the torch binding.
// Pointer members default to null (same NSDMI rationale as AttentionParams:
// bias / out_scale gate optional paths via null checks, so a partially
// packed struct must never hold garbage non-null pointers). Still an
// aggregate, still trivially copyable.
struct FP8Params {
    // Inputs: a/b are FP8 for the pre-quantized path. Scales are
    // quantization steps (device scalars).
    const void* __restrict__ a_ptr = nullptr;
    const void* __restrict__ b_ptr = nullptr;
    const void* __restrict__ bias = nullptr;
    const float* __restrict__ scale_a = nullptr;
    const float* __restrict__ scale_b = nullptr;
    const float* __restrict__ bias_scale = nullptr;
    // Output: BF16 or FP8 (E4M3). out_scale is the output quantization step
    // (FP8 output only).
    void* __restrict__ out_ptr = nullptr;
    const float* __restrict__ out_scale = nullptr;

    // Shapes. `int` covers every realistic LLM shape; the kernels promote
    // to int64 for all pointer arithmetic.
    int m, n, k;

    // Physical leading dimensions (column count, i.e. row stride) of A and
    // B. For a non-transposed operand the stride equals the contract dim;
    // for a transposed operand it is the operand's own column count. The
    // binding packs these so the kernel reads both buffers either naturally
    // or transposed depending on the LayoutA/LayoutB tags (see gemm.cuh).
    int a_ld, b_ld;
};

}  // namespace fp8
}  // namespace astrai
