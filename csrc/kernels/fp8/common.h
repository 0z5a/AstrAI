#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

// Pure POD/traits header — no .cuh/CUDA-kernel includes; raw __nv_* type
// spellings only.

namespace astrai {
namespace fp8 {

// Compile-time FP8 format: E4M3 (forward, max 448) or E5M2 (gradients,
// max 57344).
enum class FP8Format : int {
    E4M3 = 0,
    E5M2 = 1,
};

// Operand storage tags (CUTLASS-style) relative to the canonical matrices
// A [M][K] / B [K][N]: A RowMajor = [M][K] (default), A ColMajor = [K][M],
// B RowMajor = [K][N], B ColMajor = [N][K] (the nn.Linear weight). Selection
// is by type at compile time (see gemm.cuh's stage loads).
struct RowMajor {};
struct ColMajor {};

// Compile-time tile configuration, mirroring KernelTraits in the attention
// kernels: CTA tile, warp tile (WarpM x WarpN — e.g. 64x32 on the 128x128
// CTA, 32x32 on the 64x64 small CTA) and cp.async pipeline depth.
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

    // Derived geometry: warp tiles tile the CTA. The smem budget is
    // layout-aware, so it lives in Fp8GemmSmem (gemm.cuh).
    static constexpr int kWarpsM = BlockM / WarpM;
    static constexpr int kWarpsN = BlockN / WarpN;
    static constexpr int kCtaThreads = kWarpsM * kWarpsN * 32;
    static_assert(kWarpsM * WarpM == BlockM && kWarpsN * WarpN == BlockN,
                  "warp tiles must exactly tile the CTA");
    static_assert(WarpM % 16 == 0 && WarpN % 8 == 0,
                  "warp tile must be a multiple of the m16n8 MMA shape");
};

// Quantize-kernel parameter POD: float input -> FP8 with fused amax.
struct FP8QuantizeParams {
    const void* __restrict__ input_ptr = nullptr;
    void* __restrict__ output_ptr = nullptr;
    void* __restrict__ output_transposed_ptr = nullptr;  // [cols][rows]
    // Output layout: 0 = row-major only, 1 = transposed only, 2 = both from
    // a single read. Modes 1/2 produce K-contiguous operands so crosswise
    // consumers (backward grad_x / grad_w) route through the NT fast path.
    int out_layout = 0;

    const float* __restrict__ scale = nullptr;  // device multiplier
    float* __restrict__ amax = nullptr;         // raw-domain max out

    // Element count (elementwise kernel); the tiled kernel views the same
    // buffer as [rows][cols] row-major.
    int total = 0;
    int rows = 0;
    int cols = 0;
};

// Unified GEMM parameter POD, mirroring AttentionParams: one struct flows
// through the kernels; each kernel touches only the fields it needs.
struct FP8Params {
    // FP8 operands + output; scales are quantization steps (device
    // scalars). Optional bf16 bias fuses into the epilogue (fp32 add before
    // the single bf16 rounding); null disables.
    const void* __restrict__ a_ptr = nullptr;
    const void* __restrict__ b_ptr = nullptr;
    const void* __restrict__ bias_ptr = nullptr;
    void* __restrict__ out_ptr = nullptr;

    const float* __restrict__ scale = nullptr;
    // NN-swap mode (canonicalize_gemm): the kernel computes the transposed
    // problem and the epilogue scatters D[row][col] to out[col * p.m + row]
    // in the caller's [M][N] buffer. Zero in the plain orientation.
    int out_transposed = 0;
    int m, n, k;  // int covers LLM shapes; kernels promote to int64

    // Batched (bmm) geometry: grid.z steps these element strides (0
    // broadcasts the operand across batches).
    int batch = 1;
    int64_t a_batch_stride = 0;
    int64_t b_batch_stride = 0;
    int64_t out_batch_stride = 0;

    // Physical leading dims (row strides) of A and B; the binding packs
    // them so the kernel reads each buffer naturally or transposed per the
    // LayoutA/LayoutB tags.
    int a_ld, b_ld;
};

}  // namespace fp8
}  // namespace astrai
