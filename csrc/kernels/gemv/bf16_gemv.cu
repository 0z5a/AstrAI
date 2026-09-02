// Directly callable small-M BF16 GEMV primitive for decode-time linear layers.

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

template <int Rows>
__global__ void bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int n,
    int k
) {
    const int output_index = blockIdx.x;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;

    float sums[Rows] = {};
    __shared__ float warp_sums[Rows][kThreads / kWarpSize];
    // Weight row: scalar head/tail around a 16-byte-aligned uint4 middle so
    // any K is accepted while keeping 128-bit weight loads, which dominate
    // bandwidth on decode shapes. x pairs with scalar loads: it is a tiny
    // L1/L2-resident matrix, consecutive threads still touch contiguous
    // addresses, and no per-row alignment case analysis is needed.
    const __nv_bfloat16* __restrict__ wrow =
        weight + static_cast<int64_t>(output_index) * k;
    const unsigned whead_raw =
        ((16u - (reinterpret_cast<uintptr_t>(wrow) & 15u)) & 15u) >> 1;
    const int whead = static_cast<int>(min(whead_raw, static_cast<unsigned>(k)));
    const int wvecs = (k - whead) / 8;
    const int wtail_start = whead + wvecs * 8;
    const uint4* __restrict__ w4 = reinterpret_cast<const uint4*>(wrow + whead);

    // x chunks pair element-for-element with the aligned weight middle. When
    // K % 8 == 0 every x row base shares the weight alignment, so one pure
    // uint4 loop covers all rows (the production case: head/tail empty, no
    // branching inside the loop). Otherwise per-row uint4 loads are not
    // 16-byte addressable, and scalar x pairing keeps the kernel correct for
    // any K while the weight stream stays vectorized.
    if (k % 8 == 0 &&
        ((reinterpret_cast<uintptr_t>(x) + 2u * static_cast<unsigned>(whead)) & 15u) == 0u) {
        const auto* x4 = reinterpret_cast<const uint4*>(x);
        for (int v = threadIdx.x; v < wvecs; v += blockDim.x) {
            const uint4 wv_raw = w4[v];
            const auto* wv =
                reinterpret_cast<const __nv_bfloat162*>(&wv_raw);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const uint4 xv_raw =
                    x4[(static_cast<int64_t>(row) * wvecs) + v];
                const auto* xv =
                    reinterpret_cast<const __nv_bfloat162*>(&xv_raw);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sums[row] = fmaf(
                        __bfloat162float(__low2bfloat16(xv[p])),
                        __bfloat162float(__low2bfloat16(wv[p])),
                        sums[row]
                    );
                    sums[row] = fmaf(
                        __bfloat162float(__high2bfloat16(xv[p])),
                        __bfloat162float(__high2bfloat16(wv[p])),
                        sums[row]
                    );
                }
            }
        }
    } else {
        for (int v = threadIdx.x; v < wvecs; v += blockDim.x) {
            const uint4 wv_raw = w4[v];
            const __nv_bfloat16* wv_s =
                reinterpret_cast<const __nv_bfloat16*>(&wv_raw);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const __nv_bfloat16* xv =
                    x + static_cast<int64_t>(row) * k + whead + 8 * v;
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    sums[row] = fmaf(
                        __bfloat162float(xv[s]),
                        __bfloat162float(wv_s[s]),
                        sums[row]
                    );
                }
            }
        }
    }
    // Head and tail remainders: plain scalar pairing, at most 14 elements.
    for (int i = threadIdx.x; i < whead; i += blockDim.x) {
        const float wv = __bfloat162float(wrow[i]);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            sums[row] = fmaf(
                __bfloat162float(x[static_cast<int64_t>(row) * k + i]),
                wv,
                sums[row]
            );
        }
    }
    for (int i = wtail_start + threadIdx.x; i < k; i += blockDim.x) {
        const float wv = __bfloat162float(wrow[i]);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            sums[row] = fmaf(
                __bfloat162float(x[static_cast<int64_t>(row) * k + i]),
                wv,
                sums[row]
            );
        }
    }

    
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        sums[row] = warp_sum(sums[row]);
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            warp_sums[row][warp] = sums[row];
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float sum =
                lane < (kThreads / kWarpSize) ? warp_sums[row][lane] : 0.0f;
            sum = warp_sum(sum);
            if (lane == 0) {
                if (bias != nullptr) {
                    sum += __bfloat162float(bias[output_index]);
                }
                output[row * n + output_index] = __float2bfloat16_rn(sum);
            }
        }
    }
}

template <int Rows>
void launch_bf16_gemv(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_bfloat16* output,
    int n,
    int k,
    cudaStream_t stream
) {
    bf16_gemv_kernel<Rows><<<n, kThreads, 0, stream>>>(
        x, weight, bias, output, n, k
    );
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
        x.dim() == 1 || x.dim() == 2,
        "x must have shape [K] or [M, K]"
    );
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "x and weight must be contiguous");
    TORCH_CHECK(
        !x.requires_grad() && !weight.requires_grad(),
        "bf16_gemv is inference-only and does not support autograd"
    );

    const int64_t m = x.dim() == 1 ? 1 : x.size(0);
    const int64_t k = x.size(-1);
    const int64_t n = weight.size(0);
    TORCH_CHECK(
        m >= 1 && m <= 8,
        "M must be in [1, 8]"
    );
    TORCH_CHECK(weight.size(1) == k, "weight K must match x K");
    TORCH_CHECK(k > 0 && n > 0, "N and K must be positive");
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
    auto output = x.dim() == 1 ? torch::empty({n}, x.options())
                               : torch::empty({m, n}, x.options());

    // Small-N decode shapes (GQA k/v projections) cannot fill the GPU with
    // one block per output row; split K across extra blocks and reduce.
    const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* weight_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
    auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    const int n_int = static_cast<int>(n);
    const int k_int = static_cast<int>(k);
    switch (m) {
        case 1:
            launch_bf16_gemv<1>(
                x_ptr, weight_ptr, bias_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 2:
            launch_bf16_gemv<2>(
                x_ptr, weight_ptr, bias_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 3:
            launch_bf16_gemv<3>(
                x_ptr, weight_ptr, bias_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 4:
            launch_bf16_gemv<4>(
                x_ptr, weight_ptr, bias_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 5:
            launch_bf16_gemv<5>(
                x_ptr, weight_ptr, bias_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 6:
            launch_bf16_gemv<6>(
                x_ptr, weight_ptr, bias_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 7:
            launch_bf16_gemv<7>(
                x_ptr, weight_ptr, bias_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 8:
            launch_bf16_gemv<8>(
                x_ptr, weight_ptr, bias_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
    }
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
        "M in [1, 8] BF16 GEMV with FP32 accumulation and optional fused bias"
    );
}
