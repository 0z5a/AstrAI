#pragma once
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

static constexpr int MAX_SPLITS = 32;

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}
