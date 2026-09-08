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

// The format enum (binding-facing dispatch key) -> the __nv_fp8 element
// type (the kernel/traits key). Primary template undefined: an unmapped
// format is a compile error at the use site, never a silent e4m3
// fallback. Shared across the family — gemm/policy.cuh aliases this.
template <FP8Format Fmt>
struct fp8_elem;

template <>
struct fp8_elem<FP8Format::E4M3> {
    using type = __nv_fp8_e4m3;
};

template <>
struct fp8_elem<FP8Format::E5M2> {
    using type = __nv_fp8_e5m2;
};

template <FP8Format Fmt>
using fp8_elem_t = typename fp8_elem<Fmt>::type;

// Compute-capability comparison: is the device at least (major, minor)?
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
