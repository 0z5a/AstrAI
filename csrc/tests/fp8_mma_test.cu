/*
Single-kernel BF16 -> FP8 MMA -> BF16 demo for Ada (sm_89).

nvcc -I csrc -arch=sm_89 -std=c++17 -O3 --use_fast_math \
    --ptxas-options=-O3,-v csrc/tests/fp8_mma_test.cu -o fp8_mma_test \
    && ./fp8_mma_test
*/

#include "test_utils.cuh"

#include <cuda_fp8.h>

#include "../kernels/common/mma.cuh"

#include <algorithm>
#include <vector>

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = 32;

__device__ __forceinline__ unsigned pack_fp8x4(float x0, float x1, float x2,
                                                float x3) {
    __nv_fp8_e4m3 q0(x0);
    __nv_fp8_e4m3 q1(x1);
    __nv_fp8_e4m3 q2(x2);
    __nv_fp8_e4m3 q3(x3);
    return static_cast<unsigned>(q0.__x) |
           (static_cast<unsigned>(q1.__x) << 8) |
           (static_cast<unsigned>(q2.__x) << 16) |
           (static_cast<unsigned>(q3.__x) << 24);
}

__device__ __forceinline__ unsigned load_quantize_fp8x4(
    const bf16* src, float scale_inv) {
    return pack_fp8x4(__bfloat162float(src[0]) * scale_inv,
                      __bfloat162float(src[1]) * scale_inv,
                      __bfloat162float(src[2]) * scale_inv,
                      __bfloat162float(src[3]) * scale_inv);
}

__global__ void fused_bf16_fp8_mma_kernel(
    const bf16* __restrict__ a, const bf16* __restrict__ b,
    bf16* __restrict__ out, float scale_a, float scale_b) {
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int k0 = thread_in_group * 4;

    // PTX m16n8k32 A fragment: two rows, two 16-column K partitions.
    unsigned a_frag[4];
    a_frag[0] = load_quantize_fp8x4(&a[group * K + k0], 1.0f / scale_a);
    a_frag[1] = load_quantize_fp8x4(&a[(group + 8) * K + k0], 1.0f / scale_a);
    a_frag[2] = load_quantize_fp8x4(&a[group * K + k0 + 16], 1.0f / scale_a);
    a_frag[3] = load_quantize_fp8x4(&a[(group + 8) * K + k0 + 16],
                                    1.0f / scale_a);

    // B is supplied as row-major [N,K], equivalent to the col-major [K,N]
    // operand required by the MMA instruction.
    unsigned b_frag[2];
    b_frag[0] = load_quantize_fp8x4(&b[group * K + k0], 1.0f / scale_b);
    b_frag[1] = load_quantize_fp8x4(&b[group * K + k0 + 16], 1.0f / scale_b);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    astrai::mma_sync<__nv_fp8_e4m3>(acc, a_frag, b_frag, acc);

    const int col = thread_in_group * 2;
    const float output_scale = scale_a * scale_b;
    *reinterpret_cast<__nv_bfloat162*>(&out[group * N + col]) =
        __floats2bfloat162_rn(acc[0] * output_scale,
                              acc[1] * output_scale);
    *reinterpret_cast<__nv_bfloat162*>(&out[(group + 8) * N + col]) =
        __floats2bfloat162_rn(acc[2] * output_scale,
                              acc[3] * output_scale);
}

static float quantize_e4m3(float value) {
    return static_cast<float>(__nv_fp8_e4m3(value));
}

int main() {
    srand(0);
    std::vector<float> a(M * K), b(N * K), reference(M * N, 0.0f);
    std::vector<bf16> a_bf16(M * K), b_bf16(N * K), output(M * N);
    for (float& value : a) value = randf() * 4.0f;
    for (float& value : b) value = randf() * 4.0f;
    for (int i = 0; i < M * K; ++i) {
        a_bf16[i] = f2bf(a[i]);
        a[i] = bf2f(a_bf16[i]);
    }
    for (int i = 0; i < N * K; ++i) {
        b_bf16[i] = f2bf(b[i]);
        b[i] = bf2f(b_bf16[i]);
    }

    const float amax = *std::max_element(
        a.begin(), a.end(), [](float x, float y) { return fabsf(x) < fabsf(y); });
    const float bmax = *std::max_element(
        b.begin(), b.end(), [](float x, float y) { return fabsf(x) < fabsf(y); });
    const float scale_a = fabsf(amax) / 448.0f;
    const float scale_b = fabsf(bmax) / 448.0f;

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float qa = quantize_e4m3(a[row * K + k] / scale_a);
                float qb = quantize_e4m3(b[col * K + k] / scale_b);
                sum = fmaf(qa, qb, sum);
            }
            reference[row * N + col] = sum * scale_a * scale_b;
        }
    }

    bf16 *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, a_bf16.size() * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_bf16.size() * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_out, output.size() * sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(d_a, a_bf16.data(), a_bf16.size() * sizeof(bf16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_bf16.data(), b_bf16.size() * sizeof(bf16),
                          cudaMemcpyHostToDevice));

    fused_bf16_fp8_mma_kernel<<<1, 32>>>(d_a, d_b, d_out, scale_a, scale_b);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), d_out, output.size() * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = fabsf(bf2f(output[i]) - reference[i]);
        max_abs_error = fmaxf(max_abs_error, error);
        max_rel_error = fmaxf(max_rel_error,
                              error / fmaxf(fabsf(reference[i]), 1e-4f));
    }
    const bool pass = max_abs_error < 0.05f;
    print_test_header();
    print_test_row("M=16 N=8 K=32 fused BF16->E4M3 MMA", max_abs_error,
                   max_rel_error, pass);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return pass ? 0 : 1;
}
