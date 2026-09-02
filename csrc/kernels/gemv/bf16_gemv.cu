// Directly callable M=1 BF16 GEMV primitive for decode-time linear layers.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int k
) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int pairs = k / 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2 = reinterpret_cast<const __nv_bfloat162*>(weight) + row * pairs;

    float sum = 0.0f;
    for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        const __nv_bfloat162 xv = x2[pair];
        const __nv_bfloat162 wv = w2[pair];
        sum = fmaf(
            __bfloat162float(__low2bfloat16(xv)),
            __bfloat162float(__low2bfloat16(wv)),
            sum
        );
        sum = fmaf(
            __bfloat162float(__high2bfloat16(xv)),
            __bfloat162float(__high2bfloat16(wv)),
            sum
        );
    }

    sum = warp_sum(sum);
    __shared__ float warp_sums[kThreads / kWarpSize];
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < (kThreads / kWarpSize) ? warp_sums[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) {
            if (bias != nullptr) {
                sum += __bfloat162float(bias[row]);
            }
            output[row] = __float2bfloat16_rn(sum);
        }
    }
}

torch::Tensor bf16_gemv(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_object
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x and weight must be CUDA tensors");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must share device");
    TORCH_CHECK(
        x.scalar_type() == torch::kBFloat16 &&
            weight.scalar_type() == torch::kBFloat16,
        "x and weight must be bf16"
    );
    TORCH_CHECK(
        x.dim() == 1 || (x.dim() == 2 && x.size(0) == 1),
        "x must have shape [K] or [1, K]"
    );
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "x and weight must be contiguous");
    TORCH_CHECK(
        !x.requires_grad() && !weight.requires_grad(),
        "bf16_gemv is inference-only and does not support autograd"
    );

    const int64_t k = x.size(-1);
    const int64_t n = weight.size(0);
    TORCH_CHECK(weight.size(1) == k, "weight K must match x K");
    TORCH_CHECK(k > 0 && n > 0, "N and K must be positive");
    TORCH_CHECK(k % 2 == 0, "K must be even for vectorized bf16 loads");
    TORCH_CHECK(
        k <= std::numeric_limits<int>::max() &&
            n <= std::numeric_limits<int>::max(),
        "N or K exceeds the CUDA launcher limit"
    );

    torch::Tensor bias;
    const __nv_bfloat16* bias_ptr = nullptr;
    if (!bias_object.is_none()) {
        bias = bias_object.cast<torch::Tensor>();
        TORCH_CHECK(bias.is_cuda() && bias.device() == x.device(), "bias must share the CUDA device");
        TORCH_CHECK(bias.scalar_type() == torch::kBFloat16, "bias must be bf16");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == n, "bias must have shape [N]");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(!bias.requires_grad(), "bf16_gemv bias does not support autograd");
        bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
    }

    const at::cuda::OptionalCUDAGuard guard(x.device());
    const auto* properties = at::cuda::getDeviceProperties(x.device().index());
    TORCH_CHECK(properties->major >= 8, "bf16_gemv requires compute capability 8.0+");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto output = x.dim() == 1
        ? torch::empty({n}, x.options())
        : torch::empty({1, n}, x.options());

    bf16_gemv_kernel<<<static_cast<int>(n), kThreads, 0, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        bias_ptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<int>(k)
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "bf16_gemv",
        &bf16_gemv,
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        "M=1 BF16 GEMV with FP32 accumulation and optional fused bias"
    );
}
