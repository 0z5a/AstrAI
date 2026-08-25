#pragma once
// FP8 quantize device code — pure CUDA, no torch. Any float input element
// type (bf16 / fp16 / fp32) converts to E4M3 or E5M2 with a fused amax over
// the raw (unscaled) values. Mirrors the GEMM file's split: kernels take the
// FP8QuantizeParams POD, formats and input types ride on template parameters,
// and the launcher is a plain function usable from both the torch binding and
// pure C tests.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "common.h"
#include "../common/reduce.cuh"

namespace astrai {
namespace fp8 {

// Input element type traits: one element -> float, and the vectorized
// unpack of one 16-byte load into kVecElems floats.
template <typename InT>
struct quant_in_traits;

template <>
struct quant_in_traits<__nv_bfloat16> {
    static constexpr int kVecElems = 8;
    static __device__ __forceinline__ float to_float(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    static __device__ __forceinline__ void load_vec(const uint4& raw,
                                                    float* f) {
        const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            f[2 * j] =
                __bfloat162float(__ushort_as_bfloat16(w[j] & 0xffffu));
            f[2 * j + 1] = __bfloat162float(__ushort_as_bfloat16(w[j] >> 16));
        }
    }
};

template <>
struct quant_in_traits<__half> {
    static constexpr int kVecElems = 8;
    static __device__ __forceinline__ float to_float(__half v) {
        return __half2float(v);
    }
    static __device__ __forceinline__ void load_vec(const uint4& raw,
                                                    float* f) {
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 p = __half22float2(h2[j]);
            f[2 * j] = p.x;
            f[2 * j + 1] = p.y;
        }
    }
};

template <>
struct quant_in_traits<float> {
    static constexpr int kVecElems = 4;
    static __device__ __forceinline__ float to_float(float v) { return v; }
    static __device__ __forceinline__ void load_vec(const uint4& raw,
                                                    float* f) {
        f[0] = __uint_as_float(raw.x);
        f[1] = __uint_as_float(raw.y);
        f[2] = __uint_as_float(raw.z);
        f[3] = __uint_as_float(raw.w);
    }
};

// Convert one float pair to one packed fp8 pair. The stored bytes see
// value * mult (round-nearest-even + satfinite).
template <FP8Format Fmt>
__device__ __forceinline__ unsigned cvt_fp8x2(float a, float b) {
    constexpr __nv_fp8_interpretation_t kFmt =
        Fmt == FP8Format::E5M2 ? __NV_E5M2 : __NV_E4M3;
    return static_cast<unsigned>(__nv_cvt_float2_to_fp8x2(
        make_float2(a, b), __NV_SATFINITE, kFmt));
}

// Quantize kernel: float input -> FP8 (E4M3 or E5M2), fused amax over raw
// values.
template <FP8Format Fmt, typename InT>
__global__ void fp8_quantize_kernel(FP8QuantizeParams p) {
    const float mult = *p.scale;
    const auto* x = static_cast<const InT*>(p.input_ptr);
    void* x8 = p.output_ptr;
    float* amax = p.amax;
    float local_amax = 0.0f;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // Vectorized body: one 16B load -> kVecElems fp8 bytes per step (8
    // elements for 16-bit inputs, 4 for fp32). Torch allocations are >=16B
    // aligned and the binding passes freshly allocated contiguous buffers,
    // so element 0 keeps the uint4 access natural; a misaligned base
    // (contiguous view with an odd storage offset) falls back to the scalar
    // loop below via total_vec = 0.
    constexpr int kVecElems = quant_in_traits<InT>::kVecElems;
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(x) |
          reinterpret_cast<uintptr_t>(x8)) &
         15) == 0;
    const int64_t total_vec = aligned ? p.total / kVecElems : 0;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec;
         i += stride) {
        float f[kVecElems];
        quant_in_traits<InT>::load_vec(xv[i], f);
        // One 32-bit word packs two fp8x2 pairs (4 elements).
        unsigned packed[kVecElems / 4];
#pragma unroll
        for (int j = 0; j < kVecElems / 4; ++j) {
            local_amax = fmaxf(
                local_amax,
                fmaxf(fmaxf(fabsf(f[4 * j]), fabsf(f[4 * j + 1])),
                      fmaxf(fabsf(f[4 * j + 2]), fabsf(f[4 * j + 3]))));
            const unsigned lo =
                cvt_fp8x2<Fmt>(f[4 * j] * mult, f[4 * j + 1] * mult);
            const unsigned hi =
                cvt_fp8x2<Fmt>(f[4 * j + 2] * mult, f[4 * j + 3] * mult);
            packed[j] = (lo & 0xffffu) | (hi << 16);
        }
        if constexpr (kVecElems == 8)
            reinterpret_cast<uint2*>(x8)[i] =
                make_uint2(packed[0], packed[1]);
        else
            reinterpret_cast<unsigned*>(x8)[i] = packed[0];
    }
    // Scalar tail (and full fallback for misaligned bases).
    for (int64_t i = total_vec * kVecElems + blockIdx.x * blockDim.x +
                      threadIdx.x;
         i < p.total; i += stride) {
        const float v = quant_in_traits<InT>::to_float(x[i]);
        local_amax = fmaxf(local_amax, fabsf(v));
        if constexpr (Fmt == FP8Format::E5M2) {
            reinterpret_cast<__nv_fp8_e5m2*>(x8)[i] =
                __nv_fp8_e5m2(v * mult);
        } else {
            reinterpret_cast<__nv_fp8_e4m3*>(x8)[i] =
                __nv_fp8_e4m3(v * mult);
        }
    }
    if (amax) {
        local_amax = warp_reduce_max(local_amax);
        __shared__ float slots[32];
        if ((threadIdx.x & 31) == 0) slots[threadIdx.x >> 5] = local_amax;
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = 0.0f;
            for (int w = 0; w < (blockDim.x >> 5); ++w)
                v = fmaxf(v, slots[w]);
            atomic_max_float(amax, v);
        }
    }
}

template <FP8Format Fmt, typename InT>
void launch_fp8_quantize(const FP8QuantizeParams& p, cudaStream_t stream) {
    constexpr int kThreads = 256;
    // One block per 256 vectors; at least one block so the scalar tail of a
    // tiny / misaligned tensor is still covered.
    constexpr int kVecElems = quant_in_traits<InT>::kVecElems;
    int64_t blocks = (p.total / kVecElems + kThreads - 1) / kThreads;
    if (blocks < 1) blocks = 1;
    fp8_quantize_kernel<Fmt, InT><<<blocks, kThreads, 0, stream>>>(p);
}

}  // namespace fp8
}  // namespace astrai
