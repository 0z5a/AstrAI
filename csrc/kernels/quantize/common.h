#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

// Pure POD/traits header — no .cuh/CUDA-kernel includes; raw __nv_* type
// spellings only. Quantize-side declarations only: the GEMM family's
// dtype-neutral tags/POD live in gemm/common.h, the fp8 tile traits in
// gemm/policy.cuh.

namespace astrai {
namespace quant {

// Compile-time FP8 format: E4M3 (forward, max 448) or E5M2 (gradients,
// max 57344).
enum class FP8Format : int {
    E4M3 = 0,
    E5M2 = 1,
};

// Quantize output orientation: RowMajor = x8 only; Transposed = the
// [cols][rows] x8T only; Dual = both from a single read. Transposed/Dual
// produce K-contiguous operands so crosswise consumers (backward
// grad_x / grad_w) route through the NT fast path.
enum class QuantLayout : int {
    RowMajor = 0,
    Transposed = 1,
    Dual = 2,
};

// Quantize-kernel parameter POD: float input -> FP8 with fused amax.
struct QuantParams {
    const void* __restrict__ input_ptr = nullptr;
    void* __restrict__ output_ptr = nullptr;
    void* __restrict__ output_transposed_ptr = nullptr;  // [cols][rows]
    QuantLayout out_layout = QuantLayout::RowMajor;

    const float* __restrict__ scale = nullptr;  // device multiplier
    float* __restrict__ amax = nullptr;         // raw-domain max out

    // Optional delayed-scaling ring fold: when fold_ring is set, the kernel's
    // last-finishing block folds the final amax into hist[hist_idx], reduces
    // the window and publishes the next scale — replacing the host-side
    // update chain. amax then points at a persistent self-cleaning slot
    // (zeroed by the same last block) inside the caller's ring state.
    bool fold_ring = false;
    float* __restrict__ hist = nullptr;  // [hist_len] amax history window
    float* __restrict__ scale_out = nullptr;
    unsigned int* __restrict__ done = nullptr;  // block-completion counter
    int hist_len = 0;
    int hist_idx = 0;
    float fp8_max = 448.0f;   // scale = max(hist) / fp8_max / pow2_margin
    float pow2_margin = 1.0f;

    // Element count (elementwise kernel); the tiled kernel views the same
    // buffer as [rows][cols] row-major.
    int total = 0;
    int rows = 0;
    int cols = 0;
};

}  // namespace quant
}  // namespace astrai
