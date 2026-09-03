// Fused small-M BF16 SwiGLU primitive for decode-time dense MLP layers.
// One CTA per output column; each weight pair is read once and reused across
// all decode rows. Bandwidth-bound in the cold-HBM decode regime, so variant
// selection beyond the M=8 block-size rule is noise (see
// docs/developer/swiglu_benchmark.md).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

template <int Threads, int Rows>
__global__ void bf16_swiglu_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ up_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    __nv_bfloat16* __restrict__ output,
    int n,
    int k
) {
    constexpr int kWarps = Threads / kWarpSize;
    const int output_index = blockIdx.x;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int vector_count = k / 8;

    float up_sums[Rows] = {};
    float gate_sums[Rows] = {};
    __shared__ float up_warp_sums[Rows][kWarps];
    __shared__ float gate_warp_sums[Rows][kWarps];

    const auto* x4 = reinterpret_cast<const uint4*>(x);
    const auto* up4 = reinterpret_cast<const uint4*>(
        up_weight + static_cast<int64_t>(output_index) * k
    );
    const auto* gate4 = reinterpret_cast<const uint4*>(
        gate_weight + static_cast<int64_t>(output_index) * k
    );

    // Read each pair of up/gate weight chunks once per CTA, then reuse it for
    // every active decode row. The fused epilogue removes two [M, N]
    // intermediates and the standalone SiLU and multiply launches.
    for (int vector_index = threadIdx.x;
         vector_index < vector_count;
         vector_index += blockDim.x) {
        const uint4 up_raw = up4[vector_index];
        const uint4 gate_raw = gate4[vector_index];
        const auto* up_values =
            reinterpret_cast<const __nv_bfloat162*>(&up_raw);
        const auto* gate_values =
            reinterpret_cast<const __nv_bfloat162*>(&gate_raw);

#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const uint4 x_raw =
                x4[static_cast<int64_t>(row) * vector_count + vector_index];
            const auto* x_values =
                reinterpret_cast<const __nv_bfloat162*>(&x_raw);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float2 xv = __bfloat1622float2(x_values[pair]);
                const float2 uv = __bfloat1622float2(up_values[pair]);
                const float2 gv = __bfloat1622float2(gate_values[pair]);
                up_sums[row] = fmaf(xv.x, uv.x, up_sums[row]);
                up_sums[row] = fmaf(xv.y, uv.y, up_sums[row]);
                gate_sums[row] = fmaf(xv.x, gv.x, gate_sums[row]);
                gate_sums[row] = fmaf(xv.y, gv.y, gate_sums[row]);
            }
        }
    }

#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        up_sums[row] = warp_sum(up_sums[row]);
        gate_sums[row] = warp_sum(gate_sums[row]);
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            up_warp_sums[row][warp] = up_sums[row];
            gate_warp_sums[row][warp] = gate_sums[row];
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float up = lane < kWarps ? up_warp_sums[row][lane] : 0.0f;
            float gate = lane < kWarps ? gate_warp_sums[row][lane] : 0.0f;
            up = warp_sum(up);
            gate = warp_sum(gate);
            if (lane == 0) {
                // Match the public composition's BF16 rounding boundaries:
                // BF16 linear outputs, BF16 SiLU output, then BF16 multiply.
                up = round_bf16(up);
                gate = round_bf16(gate);
                const float silu = round_bf16(gate / (1.0f + expf(-gate)));
                output[static_cast<int64_t>(row) * n + output_index] =
                    __float2bfloat16_rn(up * silu);
            }
        }
    }
}

template <int Threads, int Rows>
void launch_bf16_swiglu(
    const __nv_bfloat16* x,
    const __nv_bfloat16* up_weight,
    const __nv_bfloat16* gate_weight,
    __nv_bfloat16* output,
    int n,
    int k,
    cudaStream_t stream
) {
    bf16_swiglu_kernel<Threads, Rows><<<n, Threads, 0, stream>>>(
        x, up_weight, gate_weight, output, n, k
    );
}

torch::Tensor bf16_swiglu(
    torch::Tensor x,
    torch::Tensor up_weight,
    torch::Tensor gate_weight
) {
    TORCH_CHECK(
        x.is_cuda() && up_weight.is_cuda() && gate_weight.is_cuda(),
        "x, up_weight, and gate_weight must be CUDA tensors"
    );
    TORCH_CHECK(
        x.device() == up_weight.device() && x.device() == gate_weight.device(),
        "x and weights must share a device"
    );
    TORCH_CHECK(
        x.scalar_type() == torch::kBFloat16 &&
            up_weight.scalar_type() == torch::kBFloat16 &&
            gate_weight.scalar_type() == torch::kBFloat16,
        "x and weights must be bf16"
    );
    TORCH_CHECK(
        x.dim() == 1 || x.dim() == 2,
        "x must have shape [K] or [M, K]"
    );
    TORCH_CHECK(
        up_weight.dim() == 2 && gate_weight.dim() == 2,
        "weights must have shape [N, K]"
    );
    TORCH_CHECK(
        x.is_contiguous() && up_weight.is_contiguous() &&
            gate_weight.is_contiguous(),
        "x and weights must be contiguous"
    );
    // The kernel loads all three streams as uint4; contiguous-but-offset
    // views would fault with an opaque "misaligned address" CUDA error, so
    // reject them here with an actionable message.
    TORCH_CHECK(
        (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15u) == 0u,
        "bf16_swiglu requires 16-byte aligned x (storage_offset must keep "
        "data_ptr divisible by 16); clone the tensor or use the torch path"
    );
    TORCH_CHECK(
        (reinterpret_cast<uintptr_t>(up_weight.data_ptr()) & 15u) == 0u,
        "bf16_swiglu requires 16-byte aligned up_weight (storage_offset "
        "must keep data_ptr divisible by 16); clone the tensor or use the "
        "torch path"
    );
    TORCH_CHECK(
        (reinterpret_cast<uintptr_t>(gate_weight.data_ptr()) & 15u) == 0u,
        "bf16_swiglu requires 16-byte aligned gate_weight (storage_offset "
        "must keep data_ptr divisible by 16); clone the tensor or use the "
        "torch path"
    );
    TORCH_CHECK(
        !x.requires_grad() && !up_weight.requires_grad() &&
            !gate_weight.requires_grad(),
        "bf16_swiglu is inference-only and does not support autograd"
    );

    const int64_t m = x.dim() == 1 ? 1 : x.size(0);
    const int64_t k = x.size(-1);
    const int64_t n = up_weight.size(0);
    TORCH_CHECK(m >= 1 && m <= 8, "M must be in [1, 8]");
    TORCH_CHECK(
        gate_weight.sizes() == up_weight.sizes(),
        "up_weight and gate_weight must have identical shapes"
    );
    TORCH_CHECK(up_weight.size(1) == k, "weight K must match x K");
    TORCH_CHECK(k > 0 && n > 0, "N and K must be positive");
    TORCH_CHECK(k % 8 == 0, "K must be divisible by 8");
    TORCH_CHECK(
        k <= std::numeric_limits<int>::max() &&
            n <= std::numeric_limits<int>::max(),
        "N or K exceeds the CUDA launcher limit"
    );

    const at::cuda::OptionalCUDAGuard guard(x.device());
    const auto* properties = at::cuda::getDeviceProperties(x.device().index());
    TORCH_CHECK(
        properties->major >= 8,
        "bf16_swiglu requires compute capability 8.0+"
    );
    auto stream = at::cuda::getCurrentCUDAStream();
    auto output = x.dim() == 1 ? torch::empty({n}, x.options())
                               : torch::empty({m, n}, x.options());

    const auto* x_ptr =
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* up_ptr =
        reinterpret_cast<const __nv_bfloat16*>(up_weight.data_ptr());
    const auto* gate_ptr =
        reinterpret_cast<const __nv_bfloat16*>(gate_weight.data_ptr());
    auto* output_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    const int n_int = static_cast<int>(n);
    const int k_int = static_cast<int>(k);

    // Block size 256 keeps the weight streams at the HBM bandwidth floor for
    // M in [1, 7]; M=8 halves the CTA so each thread owns more of the row
    // and the shared-memory reduction tree shrinks (measured on L20 with
    // rotated cold weights; larger CTAs only add idle warps).
    switch (m) {
        case 1:
            launch_bf16_swiglu<256, 1>(
                x_ptr, up_ptr, gate_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 2:
            launch_bf16_swiglu<256, 2>(
                x_ptr, up_ptr, gate_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 3:
            launch_bf16_swiglu<256, 3>(
                x_ptr, up_ptr, gate_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 4:
            launch_bf16_swiglu<256, 4>(
                x_ptr, up_ptr, gate_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 5:
            launch_bf16_swiglu<256, 5>(
                x_ptr, up_ptr, gate_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 6:
            launch_bf16_swiglu<256, 6>(
                x_ptr, up_ptr, gate_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 7:
            launch_bf16_swiglu<256, 7>(
                x_ptr, up_ptr, gate_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
        case 8:
            launch_bf16_swiglu<128, 8>(
                x_ptr, up_ptr, gate_ptr, output_ptr, n_int, k_int, stream.stream()
            );
            break;
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "bf16_swiglu",
        &bf16_swiglu,
        py::arg("x"),
        py::arg("up_weight"),
        py::arg("gate_weight"),
        "M in [1, 8] fused BF16 up/gate projection and SwiGLU"
    );
}
