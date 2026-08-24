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

// Unified GEMM parameter POD, mirroring AttentionParams: one struct flows
// through quantize / fused / pre-quantized kernels. Each kernel touches only
// the fields it needs; buffers are raw pointers packed by the torch binding.
struct FP8Params {
    // Inputs: a/b are BF16 for the fused (quantize-in-GEMM) path, FP8 for
    // the pre-quantized path. Scales are quantization steps (device scalars).
    const void* __restrict__ a_ptr;
    const void* __restrict__ b_ptr;
    const float* __restrict__ scale_a;
    const float* __restrict__ scale_b;

    // Output: BF16 or FP8 (E4M3). out_scale is the output quantization step
    // (FP8 output only).
    void* __restrict__ out_ptr;
    const float* __restrict__ out_scale;

    // Fused forward extras: bias (may be null) and amax slots (may be null).
    const __nv_bfloat16* __restrict__ bias;
    float* __restrict__ amax_a;
    float* __restrict__ amax_b;

    // Shapes. total is only used by the elementwise quantize kernel. `int`
    // covers every realistic LLM shape; the kernels promote to int64 for all
    // pointer arithmetic.
    int m, n, k;

    // Physical leading dimensions (column count, i.e. row stride) of A and B.
    // For a non-transposed operand the stride equals the contract dim; for a
    // transposed operand it is the operand's own column count. The binding
    // packs these so the kernel reads both buffers either naturally or
    // transposed depending on TransA/TransB.
    int a_ld, b_ld;

    int total;
};

}  // namespace fp8
}  // namespace astrai
