// CUDA bindings for the two stateless FP8 primitives.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "../common/device.cuh"
#include "gemm.cuh"
#include "quantize.cuh"

using namespace astrai::fp8;

namespace {

void check_fp8_device(const torch::Tensor& tensor) {
    static std::mutex mutex;
    static std::unordered_map<int, bool> supported;
    const int device = tensor.device().index();
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = supported.find(device);
        if (it != supported.end()) {
            TORCH_CHECK(it->second, "FP8 MMA requires compute capability 8.9+");
            return;
        }
    }
    const auto* properties = at::cuda::getDeviceProperties(device);
    const bool ok = astrai::sm_at_least(
        properties->major, properties->minor, astrai::kMinSmForFp8Major,
        astrai::kMinSmForFp8Minor);
    {
        std::lock_guard<std::mutex> lock(mutex);
        supported.emplace(device, ok);
    }
    TORCH_CHECK(ok, "FP8 MMA requires compute capability 8.9+");
}

void check_scale(const torch::Tensor& scale, const torch::Tensor& input) {
    TORCH_CHECK(scale.is_cuda() && scale.device() == input.device() &&
                    scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
                "scale must be a CUDA float32 scalar on the input device");
}

void pack_quantize(FP8QuantizeParams& p, const void* input, void* output,
                   const torch::Tensor& scale, torch::Tensor& amax,
                   int64_t total) {
    p.input_ptr = input;
    p.output_ptr = output;
    p.scale = scale.data_ptr<float>();
    p.amax = amax.data_ptr<float>();
    p.total = static_cast<int>(total);
}

void pack_gemm(FP8Params& p, const void* a, const void* b, void* output,
               const torch::Tensor& scale, int64_t m, int64_t n, int64_t k,
               int64_t a_ld, int64_t b_ld) {
    p.a_ptr = a;
    p.b_ptr = b;
    p.out_ptr = output;
    p.scale = scale.data_ptr<float>();
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.a_ld = static_cast<int>(a_ld);
    p.b_ld = static_cast<int>(b_ld);
}

template <FP8Format Fmt, int Variant>
void launch_variant(const FP8Params& p, cudaStream_t stream) {
    using LayoutA = std::conditional_t<(Variant & 2) != 0, ColMajor, RowMajor>;
    using LayoutB = std::conditional_t<(Variant & 1) != 0, ColMajor, RowMajor>;
    launch_fp8_gemm<Fmt, LayoutA, LayoutB>(p, stream);
}

template <FP8Format Fmt>
void dispatch_gemm(const FP8Params& p, cudaStream_t stream, bool trans_a,
                   bool trans_b) {
    const int variant = (static_cast<int>(trans_a) << 1) |
                        static_cast<int>(trans_b);
    switch (variant) {
        case 0: launch_variant<Fmt, 0>(p, stream); break;
        case 1: launch_variant<Fmt, 1>(p, stream); break;
        case 2: launch_variant<Fmt, 2>(p, stream); break;
        case 3: launch_variant<Fmt, 3>(p, stream); break;
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize(torch::Tensor x,
                                                  torch::Tensor scale,
                                                  int64_t fmt) {
    TORCH_CHECK(x.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 ||
                    x.scalar_type() == torch::kHalf ||
                    x.scalar_type() == torch::kFloat32,
                "x must be bf16, fp16 or fp32");
    TORCH_CHECK(fmt == static_cast<int64_t>(FP8Format::E4M3) ||
                    fmt == static_cast<int64_t>(FP8Format::E5M2),
                "unsupported quantization type: expected E4M3 (0) or E5M2 (1)");
    check_scale(scale, x);
    check_fp8_device(x);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto input = x.contiguous();
    auto output = torch::empty_like(
        input, input.options().dtype(fmt ? torch::kFloat8_e5m2
                                          : torch::kFloat8_e4m3fn));
    auto amax = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    FP8QuantizeParams p;
    pack_quantize(p, input.data_ptr(), output.data_ptr(), scale, amax,
                  input.numel());
    const bool e5m2 = fmt == static_cast<int64_t>(FP8Format::E5M2);
    if (x.scalar_type() == torch::kHalf) {
        if (e5m2)
            launch_fp8_quantize<FP8Format::E5M2, __half>(p, stream.stream());
        else
            launch_fp8_quantize<FP8Format::E4M3, __half>(p, stream.stream());
    } else if (x.scalar_type() == torch::kFloat32) {
        if (e5m2)
            launch_fp8_quantize<FP8Format::E5M2, float>(p, stream.stream());
        else
            launch_fp8_quantize<FP8Format::E4M3, float>(p, stream.stream());
    } else {
        if (e5m2)
            launch_fp8_quantize<FP8Format::E5M2, __nv_bfloat16>(
                p, stream.stream());
        else
            launch_fp8_quantize<FP8Format::E4M3, __nv_bfloat16>(
                p, stream.stream());
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return {output, amax};
}

torch::Tensor mm_fp8(torch::Tensor a, torch::Tensor b, torch::Tensor scale,
                     int64_t trans_a, int64_t trans_b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn ||
                    a.scalar_type() == torch::kFloat8_e5m2,
                "a and b must be fp8");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must share format");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.device() == b.device(), "a and b must share device");
    check_scale(scale, a);
    check_fp8_device(a);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    const bool ta = trans_a != 0;
    const bool tb = trans_b != 0;
    const int64_t a_ld = a_c.size(1);
    const int64_t b_ld = b_c.size(1);
    const int64_t m = ta ? a_c.size(1) : a_c.size(0);
    const int64_t k = ta ? a_c.size(0) : a_c.size(1);
    const int64_t n = tb ? b_c.size(0) : b_c.size(1);
    TORCH_CHECK(k == (tb ? b_c.size(1) : b_c.size(0)), "inner dim mismatch");
    auto output = torch::empty({m, n}, a_c.options().dtype(torch::kBFloat16));
    FP8Params p;
    pack_gemm(p, a_c.data_ptr(), b_c.data_ptr(), output.data_ptr(), scale, m, n,
              k, a_ld, b_ld);
    if (a.scalar_type() == torch::kFloat8_e4m3fn)
        dispatch_gemm<FP8Format::E4M3>(p, stream.stream(), ta, tb);
    else
        dispatch_gemm<FP8Format::E5M2>(p, stream.stream(), ta, tb);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize, py::arg("x"), py::arg("scale"),
          py::arg("fmt"));
    m.def("mm_fp8", &mm_fp8, py::arg("a"), py::arg("b"), py::arg("scale"),
          py::arg("trans_a") = 0, py::arg("trans_b") = 0);
}
