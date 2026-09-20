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

// FP8 formats are the raw element types (__nv_fp8_e4m3 / __nv_fp8_e5m2);
// the bindings name them directly from the output dtype — no format enum.
inline bool sm_at_least(int device_major, int device_minor, int major,
                        int minor) {
    return device_major > major ||
           (device_major == major && device_minor >= minor);
}

// FP8 tensor-core MMA (`mma.sync.aligned.m16n8k32` with fp8 inputs) exists
// on Ada (sm_89) and Hopper (sm_90+); sm_80 has no fp8 instructions. The
// bindings check it at their entry (getDeviceProperties is ATen-cached).
inline constexpr int kMinSmForFp8Major = 8;
inline constexpr int kMinSmForFp8Minor = 9;

// Quantize output orientation: RowMajor = x8 only; Transposed = the
// [cols][rows] x8T only; Dual = both from a single read. Transposed/Dual
// produce K-contiguous operands so crosswise consumers (backward
// grad_x / grad_w) route through the NT fast path.
enum class QuantLayout : int {
    RowMajor = 0,
    Transposed = 1,
    Dual = 2,
};

// Ring-fold scratch lines: the fused amax is RMW-spread across this many
// float slots (block id mod kFoldSlots) instead of one contended address —
// a 49k-block grid otherwise serializes every block completion on a single
// L2 atomic (ncu: ~6% of the tiled dual kernel). The last-finishing block
// folds the slots, publishes the scale and re-zeroes them (self-cleaning,
// same protocol as the old single slot).
inline constexpr int kFoldSlots = 32;

// Quantize-kernel parameter POD: float input -> FP8 with fused amax.
struct QuantParams {
    const void* __restrict__ input_ptr = nullptr;
    void* __restrict__ output_ptr = nullptr;
    void* __restrict__ output_transposed_ptr = nullptr;  // [cols][rows]

    const float* __restrict__ scale = nullptr;  // device multiplier
    float* __restrict__ amax = nullptr;         // raw-domain max out

    // Optional delayed-scaling ring fold: when fold_ring is set, the kernel's
    // last-finishing block folds the final amax into hist[hist_idx], reduces
    // the window and publishes the next scale — replacing the host-side
    // update chain. Blocks RMW their block amax into amax_scratch[block id
    // mod kFoldSlots] (one contended address serializes every completion);
    // the last block folds the scratch, zeroes it and publishes.
    bool fold_ring = false;
    float* __restrict__ hist = nullptr;  // [hist_len] amax history window
    float* __restrict__ scale_out = nullptr;
    // The published scale's correctly rounded reciprocal (__frcp_rn), in the
    // ring slot the next quantize reads as its multiplier: publishing both
    // here is what keeps the host out of the per-step scale chain.
    float* __restrict__ scale_recip_out = nullptr;
    float* __restrict__ amax_scratch = nullptr;  // [kFoldSlots] RMW lines
    unsigned int* __restrict__ done = nullptr;   // block-completion counter
    int hist_len = 0;
    int hist_idx = 0;
    float fp8_max = 448.0f;   // scale = max(hist) / fp8_max / pow2_margin
    float pow2_margin = 1.0f;

    // The tiled kernel views the buffer as [rows][cols] row-major.
    int rows = 0;
    int cols = 0;

    // Unified-strided experiment: element strides of output_ptr in (row,
    // col) input coordinates — the row-major placement is (cols, 1). The
    // transposed copy's placement is NOT this pair's swap (that only holds
    // on square shapes); it is the canonical consumer contract (1, rows),
    // derived from p.rows inside the kernel — no second pair rides the POD.
    int out_row_stride = 0;
    int out_col_stride = 0;
};

}  // namespace quant
}  // namespace astrai
