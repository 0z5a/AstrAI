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

// Tiled transpose quantize (out_layout 1/2): reads the [rows][cols]
// row-major input once and writes the fp8 bytes transposed ([cols][rows],
// so the contract dim lands K-contiguous for NT GEMM operands) and, in
// mode 2, the plain row-major copy too. A 32x32 tile stages through shared
// memory: input-row-major loads and output writes both stay coalesced, and
// the byte-wide staging is conflict-free — the +4 pad makes the store
// stride 9 (words) coprime with the 32 banks and the load is a 32-byte
// broadcast segment. A 64x64 split-half variant (16 elems/thread, paired
// 2-byte scatter stores) measured +21% on L2-resident shapes but -3..5%
// on the DRAM-bound ones that carry the training traffic (occupancy and
// memory-level parallelism, not instruction count, gate the DRAM regime);
// weighted by the real step's mix the two tie, so the simpler tile stays.
template <FP8Format Fmt, typename InT>
__global__ void fp8_quantize_tiled_kernel(FP8QuantizeParams p) {
    constexpr int kTile = 32;
    __shared__ uint8_t tile[kTile][kTile + 4];
    const float mult = *p.scale;
    const auto* x = static_cast<const InT*>(p.input_ptr);
    const int r0 = blockIdx.y * kTile;
    const int c0 = blockIdx.x * kTile;
    const int r = r0 + threadIdx.y * 4;
    const int c = c0 + threadIdx.x;

    uint8_t q[4];
    float local_amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        q[j] = 0;
        if (r + j < p.rows && c < p.cols) {
            const float v =
                quant_in_traits<InT>::to_float(x[(int64_t)(r + j) * p.cols + c]);
            local_amax = fmaxf(local_amax, fabsf(v));
            if constexpr (Fmt == FP8Format::E5M2)
                q[j] = __nv_fp8_e5m2(v * mult).__x;
            else
                q[j] = __nv_fp8_e4m3(v * mult).__x;
        }
    }
    if (p.out_layout == 2) {
        uint8_t* out = static_cast<uint8_t*>(p.output_ptr);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (r + j < p.rows && c < p.cols)
                out[(int64_t)(r + j) * p.cols + c] = q[j];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) tile[threadIdx.x][threadIdx.y * 4 + j] = q[j];
    __syncthreads();
    // Transposed scatter: output element (c, r) lives at c * rows + r; r
    // tracks threadIdx.x so each warp writes one contiguous run. The read
    // swaps the staging indices — tile[col][row] was written, so the value
    // for input (r0+tx, c0+ty*4+j) sits at tile[ty*4+j][tx].
    uint8_t* out_t = static_cast<uint8_t*>(p.output_transposed_ptr);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int oc = c0 + threadIdx.y * 4 + j;
        if (oc < p.cols && r0 + threadIdx.x < p.rows)
            out_t[(int64_t)oc * p.rows + r0 + threadIdx.x] =
                tile[threadIdx.y * 4 + j][threadIdx.x];
    }
    if (p.amax) {
        local_amax = warp_reduce_max(local_amax);
        __shared__ float slots[8];
        // blockDim.x is 32, so warp id == threadIdx.y; only complete warps
        // exist (blockDim.y == 8).
        if (threadIdx.x == 0) slots[threadIdx.y] = local_amax;
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            float v = 0.0f;
            for (int w = 0; w < (int)blockDim.y; ++w) v = fmaxf(v, slots[w]);
            atomic_max_float(p.amax, v);
        }
    }
}

template <FP8Format Fmt, typename InT>
void launch_fp8_quantize_tiled(const FP8QuantizeParams& p,
                               cudaStream_t stream) {
    const dim3 grid((p.cols + 31) / 32, (p.rows + 31) / 32);
    if (grid.x == 0 || grid.y == 0) return;
    fp8_quantize_tiled_kernel<Fmt, InT>
        <<<grid, dim3(32, 8), 0, stream>>>(p);
}

}  // namespace fp8
}  // namespace astrai
