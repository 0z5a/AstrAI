#pragma once
// FP8 quantize device code — pure CUDA, no torch: kernels take the
// FP8QuantizeParams POD, format and input type ride on template parameters,
// and the launcher is shared by the torch binding and the C tests.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "common.h"
#include "../common/reduce.cuh"

namespace astrai {
namespace fp8 {

// Input element type traits: one element -> float, and the unpack of one
// 16-byte load into kVecElems floats.
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
        const __nv_bfloat162* b2 =
            reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 p = __bfloat1622float2(b2[j]);
            f[2 * j] = p.x;
            f[2 * j + 1] = p.y;
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
        const unsigned* w = reinterpret_cast<const unsigned*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) f[j] = __uint_as_float(w[j]);
    }
};

// One float -> one fp8 byte (round-nearest-even + satfinite).
template <FP8Format Fmt>
__device__ __forceinline__ uint8_t cvt_fp8(float v) {
    if constexpr (Fmt == FP8Format::E5M2)
        return __nv_fp8_e5m2(v).__x;
    else
        return __nv_fp8_e4m3(v).__x;
}

// One float pair -> one packed fp8x2 word (round-nearest-even + satfinite).
template <FP8Format Fmt>
__device__ __forceinline__ unsigned cvt_fp8x2(float a, float b) {
    constexpr __nv_fp8_interpretation_t kFmt =
        Fmt == FP8Format::E5M2 ? __NV_E5M2 : __NV_E4M3;
    return static_cast<unsigned>(__nv_cvt_float2_to_fp8x2(
        make_float2(a, b), __NV_SATFINITE, kFmt));
}

// Block-wide amax reduce -> one atomic per block: warp-reduce, park one
// value per warp, thread 0 folds. kWarps must cover the block's warp count.
template <int kWarps>
__device__ __forceinline__ void publish_amax(float* amax, float v) {
    v = warp_reduce_max(v);
    __shared__ float slots[kWarps];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if ((tid & 31) == 0) slots[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
#pragma unroll
        for (int w = 1; w < kWarps; ++w) v = fmaxf(v, slots[w]);
        atomic_max_float(amax, v);
    }
}

// Elementwise quantize kernel (out_layout 0): vectorized 16B loads -> fp8
// stores, fused amax over raw values.
template <FP8Format Fmt, typename InT>
__global__ void fp8_quantize_kernel(FP8QuantizeParams p) {
    const float mult = *p.scale;
    const auto* x = static_cast<const InT*>(p.input_ptr);
    uint8_t* x8 = static_cast<uint8_t*>(p.output_ptr);
    float local_amax = 0.0f;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // One 16B load -> kVecElems bytes per step. Torch allocations are >=16B
    // aligned, so element 0 keeps the uint4 access natural; a misaligned
    // base (odd storage offset view) falls to the scalar tail via
    // total_vec = 0.
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
            reinterpret_cast<uint2*>(x8)[i] = make_uint2(packed[0], packed[1]);
        else
            reinterpret_cast<unsigned*>(x8)[i] = packed[0];
    }
    // Scalar tail (and full fallback for misaligned bases).
    for (int64_t i = total_vec * kVecElems + blockIdx.x * blockDim.x +
                      threadIdx.x;
         i < p.total; i += stride) {
        const float v = quant_in_traits<InT>::to_float(x[i]);
        local_amax = fmaxf(local_amax, fabsf(v));
        x8[i] = cvt_fp8<Fmt>(v * mult);
    }
    if (p.amax) publish_amax<8>(p.amax, local_amax);
}

// Tiled transpose quantize (out_layout 1/2): reads the [rows][cols] input
// once and writes the fp8 bytes transposed ([cols][rows], so the contract
// dim lands K-contiguous for NT GEMM operands) and, in mode 2, the row-major
// copy too. A 32x32 tile stages through shared memory: loads and writes
// both stay coalesced, and the byte-wide staging is conflict-free — the +4
// pad makes the store stride 9 words (coprime with the 32 banks) and the
// read is a 32-byte broadcast segment. (A 64x64 split-half variant measured
// +21% L2-resident but -3..5% DRAM-bound; the real step mix ties, so the
// simpler tile stays.)
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
            q[j] = cvt_fp8<Fmt>(v * mult);
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
    // tracks threadIdx.x so each warp writes one contiguous run. tile was
    // written as tile[col][row], so input (r0+tx, c0+ty*4+j) reads back
    // from tile[ty*4+j][tx].
    uint8_t* out_t = static_cast<uint8_t*>(p.output_transposed_ptr);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int oc = c0 + threadIdx.y * 4 + j;
        if (oc < p.cols && r0 + threadIdx.x < p.rows)
            out_t[(int64_t)oc * p.rows + r0 + threadIdx.x] =
                tile[threadIdx.y * 4 + j][threadIdx.x];
    }
    if (p.amax) publish_amax<8>(p.amax, local_amax);
}

// Unified quantize launcher: Tiled selects the transpose kernel (out_layout
// 1/2) over the vectorized elementwise one.
template <FP8Format Fmt, typename InT, bool Tiled = false>
void launch_fp8_quantize(const FP8QuantizeParams& p, cudaStream_t stream) {
    if constexpr (Tiled) {
        const dim3 grid((p.cols + 31) / 32, (p.rows + 31) / 32);
        if (grid.x == 0 || grid.y == 0) return;
        fp8_quantize_tiled_kernel<Fmt, InT>
            <<<grid, dim3(32, 8), 0, stream>>>(p);
    } else {
        constexpr int kThreads = 256;
        constexpr int kVecElems = quant_in_traits<InT>::kVecElems;
        // One block per 256 vectors; at least one block so a tiny or
        // misaligned tensor's scalar tail is still covered.
        int64_t blocks = (p.total / kVecElems + kThreads - 1) / kThreads;
        if (blocks < 1) blocks = 1;
        fp8_quantize_kernel<Fmt, InT><<<blocks, kThreads, 0, stream>>>(p);
    }
}

}  // namespace fp8
}  // namespace astrai
