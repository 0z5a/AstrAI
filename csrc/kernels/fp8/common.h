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
// and the MMA PTX mnemonic; the remaining parameters shape the CTA tile, the
// warp tile (WarpM x WarpN — e.g. 64x32 on the 128x128 CTA, or 32x32 on the
// cuBLAS-style 64x64 small CTA that lifts small-shape occupancy) and the
// cp.async pipeline depth.
template <FP8Format Fmt, int BlockM, int BlockN, int K, int Stages,
          int WarpM = 64, int WarpN = 32>
struct Fp8GemmTraits {
    static constexpr FP8Format kFormat = Fmt;
    static constexpr int kBlockM = BlockM;
    static constexpr int kBlockN = BlockN;
    static constexpr int kK = K;
    static constexpr int kStages = Stages;
    static constexpr int kWarpM = WarpM;
    static constexpr int kWarpN = WarpN;
    static constexpr bool kIsE5M2 = (Fmt == FP8Format::E5M2);
    static constexpr __nv_fp8_interpretation_t kNvFormat =
        kIsE5M2 ? __NV_E5M2 : __NV_E4M3;
    static constexpr float kFp8Max = kIsE5M2 ? 57344.0f : 448.0f;

    // Derived launch geometry: WarpM x WarpN warp tiles tile the CTA. The
    // shared-memory budget is layout-aware (crosswise operands add K-major
    // staging + a canonical buffer), so it lives in Fp8GemmSmem in gemm.cuh
    // together with the resident-CTA hint for __launch_bounds__.
    static constexpr int kWarpsM = BlockM / WarpM;
    static constexpr int kWarpsN = BlockN / WarpN;
    static constexpr int kCtaThreads = kWarpsM * kWarpsN * 32;
    static_assert(kWarpsM * WarpM == BlockM && kWarpsN * WarpN == BlockN,
                  "warp tiles must exactly tile the CTA");
    static_assert(WarpM % 16 == 0 && WarpN % 8 == 0,
                  "warp tile must be a multiple of the m16n8 MMA shape");
};

// Quantize-kernel parameter POD: float input (bf16 / fp16 / fp32) -> FP8
// with fused amax.
struct FP8QuantizeParams {
    // Float input and FP8 output buffers; scale is the quantization
    // multiplier (device scalar). amax (may be null) is zero-initialized by
    // the binding and receives the raw-domain absolute maximum.
    const void* __restrict__ input_ptr = nullptr;
    void* __restrict__ output_ptr = nullptr;

    const float* __restrict__ scale = nullptr;
    float* __restrict__ amax = nullptr;

    // Element count (only the elementwise quantize kernel uses it).
    int total = 0;
};

// Unified GEMM parameter POD, mirroring AttentionParams: one struct flows
// through the pre-quantized GEMM kernels. Each kernel touches only the
// fields it needs; buffers are raw pointers packed by the torch binding.
// Pointer members default to null so optional paths cannot hold garbage.
struct FP8Params {
    // Inputs: a/b are FP8 for the pre-quantized path. Scales are
    // quantization steps (device scalars).
    // Optional bf16 bias broadcast over output rows (fused into the epilogue
    // before the bf16 rounding, so it adds in fp32 — one rounding fewer than
    // the separate out + bias elementwise kernel it replaces). Null disables.
    const void* __restrict__ a_ptr = nullptr;
    const void* __restrict__ b_ptr = nullptr;
    const void* __restrict__ bias_ptr = nullptr;
    void* __restrict__ out_ptr = nullptr;

    const float* __restrict__ scale = nullptr;
    // Shapes. `int` covers every realistic LLM shape; the kernels promote
    // to int64 for all pointer arithmetic.
    int m, n, k;

    // Batched (bmm) geometry: grid.z slices step the operand/output pointers
    // by these element strides (0 broadcasts the operand across batches).
    int batch = 1;
    int64_t a_batch_stride = 0;
    int64_t b_batch_stride = 0;
    int64_t out_batch_stride = 0;

    // Physical leading dimensions (column count, i.e. row stride) of A and
    // B. For a non-transposed operand the stride equals the contract dim;
    // for a transposed operand it is the operand's own column count. The
    // binding packs these so the kernel reads both buffers either naturally
    // or transposed depending on the LayoutA/LayoutB tags (see gemm.cuh).
    int a_ld, b_ld;
};

}  // namespace fp8
}  // namespace astrai
