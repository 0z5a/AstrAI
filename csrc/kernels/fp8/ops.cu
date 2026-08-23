// FP8 GEMM torch binding: tensor validation, FP8Params packing, template
// dispatch and pybind. Device code lives in gemm.cuh (pure CUDA) —
// mirroring the attn_*.cu / attn_*_mma.cuh split of the attention kernels.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "gemm.cuh"
#include "../common/device.cuh"

namespace {

// FP8Format / FP8Params live in the global namespace (common.h); the
// launchers live in fp8:: (gemm.cuh).

void check_fp8_device(const torch::Tensor& tensor) {
    static std::mutex mutex;
    static std::unordered_map<int, bool> supported;
    const int device = tensor.device().index();
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto cached = supported.find(device);
        if (cached != supported.end()) {
            TORCH_CHECK(cached->second,
                        "fused FP8 MMA requires compute capability 8.9 or newer");
            return;
        }
    }

    const auto* properties = at::cuda::getDeviceProperties(device);
    const bool is_supported =
        astrai::sm_at_least(properties->major, properties->minor,
                            astrai::kMinSmForFp8Major,
                            astrai::kMinSmForFp8Minor);
    {
        std::lock_guard<std::mutex> lock(mutex);
        supported.emplace(device, is_supported);
    }
    TORCH_CHECK(is_supported,
                "fused FP8 MMA requires compute capability 8.9 or newer");
}

void check_scale(const torch::Tensor& scale, const torch::Tensor& input,
                 const char* name) {
    TORCH_CHECK(scale.is_cuda() && scale.device() == input.device() &&
                    scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
                name, " must be a CUDA float32 scalar on the input device");
}

// ---- FP8Params packing (mirrors attention/entry_utils.cuh pack_* helpers) ----

void pack_gemm_params(FP8Params& p, const void* a, const void* b, void* out,
                      const torch::Tensor& sa, const torch::Tensor& sb,
                      const torch::Tensor* out_scale, int64_t m, int64_t n,
                      int64_t k) {
    p.a_ptr = a;
    p.b_ptr = b;
    p.out_ptr = out;
    p.scale_a = sa.data_ptr<float>();
    p.scale_b = sb.data_ptr<float>();
    p.out_scale = out_scale ? out_scale->data_ptr<float>() : nullptr;
    p.bias = nullptr;
    p.amax_a = nullptr;
    p.amax_b = nullptr;
    p.m = m;
    p.n = n;
    p.k = k;
    p.total = 0;
}

void pack_quantize_params(FP8Params& p, const void* x, void* x8,
                          const torch::Tensor& scale, torch::Tensor* amax,
                          int64_t total) {
    p.a_ptr = x;
    p.b_ptr = nullptr;
    p.out_ptr = x8;
    p.scale_a = scale.data_ptr<float>();
    p.scale_b = nullptr;
    p.out_scale = nullptr;
    p.bias = nullptr;
    p.amax_a = amax ? amax->data_ptr<float>() : nullptr;
    p.amax_b = nullptr;
    p.m = p.n = p.k = 0;
    p.total = total;
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> quantize_bf16(torch::Tensor x,
                                                       torch::Tensor scale,
                                                       int64_t fmt) {
    // BF16 -> FP8 quantize with fused amax. fmt: 0 = E4M3, 1 = E5M2.
    // Returns (x8, amax); the caller never clears amax (zero-initialized here).
    TORCH_CHECK(x.is_cuda() && scale.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    check_scale(scale, x, "scale");
    check_fp8_device(x);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();
    auto x8 = torch::empty_like(
        x_c, x_c.options().dtype(fmt ? torch::kFloat8_e5m2
                                     : torch::kFloat8_e4m3fn));
    auto amax = torch::zeros({1}, x_c.options().dtype(torch::kFloat32));
    FP8Params p;
    pack_quantize_params(p, x_c.data_ptr(), x8.data_ptr(), scale, &amax,
                         x_c.numel());
    if (fmt) {
        fp8::launch_fp8_quantize<FP8Format::E5M2>(p, stream.stream());
    } else {
        fp8::launch_fp8_quantize<FP8Format::E4M3>(p, stream.stream());
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return {x8, amax};
}

torch::Tensor mm_fp8(torch::Tensor a, torch::Tensor b, torch::Tensor sa,
                     torch::Tensor sb, int64_t out_dtype,
                     c10::optional<torch::Tensor> out_scale) {
    // Pre-quantized FP8 GEMM: out = a @ b^T * (sa * sb), FP32 accumulation.
    // out_dtype: 0 = BF16 (default), 1 = FP8 E4M3 (requires out_scale, the
    // quantization step for the output — mirrors torch._scaled_mm's
    // out_dtype / scale_result). Both operands share the same FP8 format.
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn ||
                    a.scalar_type() == torch::kFloat8_e5m2,
                "a and b must be fp8 (e4m3fn or e5m2)");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                "a and b must share the same fp8 format");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.size(1) == b.size(1), "inner dim mismatch");
    check_scale(sa, a, "sa");
    check_scale(sb, a, "sb");
    check_fp8_device(a);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    int64_t m = a_c.size(0), k = a_c.size(1), n = b_c.size(0);
    const bool out_fp8 = (out_dtype == 1);
    TORCH_CHECK(out_dtype == 0 || out_fp8,
                "out_dtype must be 0 (bf16) or 1 (fp8 e4m3)");
    torch::Tensor os;
    if (out_fp8) {
        TORCH_CHECK(out_scale.has_value(), "fp8 output requires out_scale");
        os = out_scale.value();
        check_scale(os, a, "out_scale");
    }
    auto out = torch::empty(
        {m, n},
        out_fp8 ? a_c.options().dtype(torch::kFloat8_e4m3fn)
                : a_c.options().dtype(torch::kBFloat16));
    FP8Params p;
    pack_gemm_params(p, a_c.data_ptr(), b_c.data_ptr(), out.data_ptr(), sa, sb,
                     out_fp8 ? &os : nullptr, m, n, k);
    if (a.scalar_type() == torch::kFloat8_e4m3fn) {
        if (out_fp8) {
            fp8::launch_fp8_gemm<FP8Format::E4M3, true>(p, stream.stream());
        } else {
            fp8::launch_fp8_gemm<FP8Format::E4M3>(p, stream.stream());
        }
    } else {
        if (out_fp8) {
            fp8::launch_fp8_gemm<FP8Format::E5M2, true>(p, stream.stream());
        } else {
            fp8::launch_fp8_gemm<FP8Format::E5M2>(p, stream.stream());
        }
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_forward_fp8(
    torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor sx,
    torch::Tensor sw, int64_t fmt) {
    // Pure FP8 forward: quantize x/w (fmt: 0 = E4M3, 1 = E5M2), then the
    // pre-quantized GEMM; the dequantized BF16 output gets the bias added.
    // amax_x / amax_w come from the quantize kernels (zero-initialized here).
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                    w.scalar_type() == torch::kBFloat16,
                "x and w must be bf16");
    TORCH_CHECK(x.device() == w.device(), "x and w must be on the same device");
    check_scale(sx, x, "sx");
    check_scale(sw, x, "sw");
    check_fp8_device(x);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_c = x.reshape({-1, w.size(1)}).contiguous();   // [M, K]
    auto w_c = w.contiguous();                            // [N, K]
    int64_t m = x_c.size(0), k = x_c.size(1), n = w_c.size(0);
    TORCH_CHECK(w_c.dim() == 2 && w_c.size(1) == k, "inner dim mismatch");
    const bool has_bias = bias.defined() && bias.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda() && bias.device() == x.device() &&
                        bias.scalar_type() == torch::kBFloat16 &&
                        bias.numel() == n,
                    "bias must be CUDA bf16 with shape [N]");
    }
    const auto f8opt = fmt ? torch::kFloat8_e5m2 : torch::kFloat8_e4m3fn;
    auto x8 = torch::empty({m, k}, x_c.options().dtype(f8opt));
    auto w8 = torch::empty({n, k}, x_c.options().dtype(f8opt));
    auto amax_x = torch::zeros({1}, x.options().dtype(torch::kFloat32));
    auto amax_w = torch::zeros({1}, x.options().dtype(torch::kFloat32));
    auto out = torch::empty({m, n}, x_c.options());

    auto quantize = [&](const torch::Tensor& src, torch::Tensor& dst,
                        const torch::Tensor& scale, torch::Tensor* amax) {
        FP8Params qp;
        pack_quantize_params(qp, src.data_ptr(), dst.data_ptr(), scale, amax,
                             src.numel());
        if (fmt) {
            fp8::launch_fp8_quantize<FP8Format::E5M2>(qp, stream.stream());
        } else {
            fp8::launch_fp8_quantize<FP8Format::E4M3>(qp, stream.stream());
        }
    };
    quantize(x_c, x8, sx, &amax_x);
    quantize(w_c, w8, sw, &amax_w);

    FP8Params p;
    pack_gemm_params(p, x8.data_ptr(), w8.data_ptr(), out.data_ptr(), sx, sw,
                     nullptr, m, n, k);
    if (fmt) {
        fp8::launch_fp8_gemm<FP8Format::E5M2>(p, stream.stream());
    } else {
        fp8::launch_fp8_gemm<FP8Format::E4M3>(p, stream.stream());
    }
    C10_CUDA_CHECK(cudaGetLastError());

    std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end() - 1);
    shape.push_back(n);
    auto out_r = out.reshape(shape);
    if (has_bias) out_r = out_r + bias;
    return {out_r, amax_x, amax_w};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
linear_backward_fp8(torch::Tensor g, torch::Tensor x, torch::Tensor w,
                    std::vector<int64_t> masks, torch::Tensor sg,
                    torch::Tensor sw, torch::Tensor sx, int64_t fmt) {
    // Pre-quantized FP8 backward: grad is quantized once (E4M3 or E5M2 per
    // `fmt`), then dX / dW run as FP8 tensor-core GEMMs sharing g8.
    // Returns (grad_input, grad_weight, grad_bias, amax_g).
    TORCH_CHECK(g.is_cuda() && x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16 &&
                    x.scalar_type() == torch::kBFloat16 &&
                    w.scalar_type() == torch::kBFloat16,
                "g, x, and w must be bf16");
    TORCH_CHECK(g.device() == x.device() && g.device() == w.device(),
                "g, x, and w must be on the same device");
    TORCH_CHECK(masks.size() == 3, "masks must contain three values");
    check_fp8_device(g);
    const at::cuda::OptionalCUDAGuard guard(g.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto g_c = g.reshape({-1, w.size(0)}).contiguous();   // [M, N]
    auto x_c = x.reshape({-1, x.size(-1)}).contiguous();  // [M, K]
    auto w_c = w.contiguous();                            // [N, K]
    int64_t m = g_c.size(0), n = w_c.size(0), k = w_c.size(1);
    TORCH_CHECK(x_c.size(0) == m && x_c.size(1) == k && g_c.size(1) == n,
                "backward shape mismatch");

    auto grad_input = torch::empty_like(x);
    auto grad_weight = torch::empty_like(w);
    auto grad_bias = torch::empty({0}, g.options());
    auto amax_g = torch::zeros({1}, g.options().dtype(torch::kFloat32));
    auto f8opt = fmt ? g.options().dtype(torch::kFloat8_e5m2)
                     : g.options().dtype(torch::kFloat8_e4m3fn);

    auto quantize = [&](const torch::Tensor& src, torch::Tensor& dst,
                        const torch::Tensor& scale, torch::Tensor* amax) {
        FP8Params qp;
        pack_quantize_params(qp, src.data_ptr(), dst.data_ptr(), scale, amax,
                             src.numel());
        if (fmt) {
            fp8::launch_fp8_quantize<FP8Format::E5M2>(qp, stream.stream());
        } else {
            fp8::launch_fp8_quantize<FP8Format::E4M3>(qp, stream.stream());
        }
    };
    // Explicit-transpose backward: the gradient/activation tensors keep their
    // natural row-major layout, which the GEMM consumes transposed (W is
    // [N,K] but dX contracts over N; x is [M,K] and g is [M,N] for dW), so
    // the fp8 operands are transposed once and run through the fast non-trans
    // pre-quantized GEMM. g is quantized once (amax_g measured here); its
    // transpose is derived from the same g8 so both GEMMs share the value.
    auto pq_n = [&](const torch::Tensor& a8, const torch::Tensor& b8,
                    torch::Tensor& out, const torch::Tensor& sa,
                    const torch::Tensor& sb, int64_t mm, int64_t nn,
                    int64_t kk) {
        FP8Params gp;
        pack_gemm_params(gp, a8.data_ptr(), b8.data_ptr(), out.data_ptr(), sa,
                         sb, nullptr, mm, nn, kk);
        if (fmt) {
            fp8::launch_fp8_gemm<FP8Format::E5M2>(gp, stream.stream());
        } else {
            fp8::launch_fp8_gemm<FP8Format::E4M3>(gp, stream.stream());
        }
    };

    torch::Tensor g8;
    if (masks[0] || masks[1]) {
        g8 = torch::empty({m, n}, f8opt);
        quantize(g_c, g8, sg, &amax_g);
    }
    // dX = g @ W: A = g8 [M,N] natural; B = W^T [K,N] (w8 transposed in fp8).
    if (masks[0]) {
        auto w8 = torch::empty({n, k}, f8opt);
        quantize(w_c, w8, sw, nullptr);
        auto w8T = w8.transpose(0, 1).contiguous();  // [K, N]
        auto grad_input_2d = grad_input.reshape({m, k});
        pq_n(g8, w8T, grad_input_2d, sg, sw, m, k, n);
    }
    // dW = g^T @ x: A = g^T [N,M] (g8 transposed); B = x^T [K,M].
    if (masks[1]) {
        auto g8T = g8.transpose(0, 1).contiguous();  // [N, M]
        auto x8 = torch::empty({m, k}, f8opt);
        quantize(x_c, x8, sx, nullptr);
        auto x8T = x8.transpose(0, 1).contiguous();  // [K, M]
        pq_n(g8T, x8T, grad_weight, sg, sx, n, k, m);
    }
    if (!masks[0] && !masks[1]) {
        amax_g.copy_(g_c.abs().amax().to(torch::kFloat32));
    }
    C10_CUDA_CHECK(cudaGetLastError());
    if (masks[2]) grad_bias = g_c.sum(0).to(g.scalar_type());
    return {grad_input, grad_weight, grad_bias, amax_g};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_bf16", &quantize_bf16, py::arg("x"), py::arg("scale"),
          py::arg("fmt"),
          "BF16 to FP8 (E4M3/E5M2) quantize with fused amax; returns (x8, amax)");
    m.def("mm_fp8", &mm_fp8, py::arg("a"), py::arg("b"), py::arg("sa"),
          py::arg("sb"), py::arg("out_dtype") = 0,
          py::arg("out_scale") = py::none(),
          "Pre-quantized FP8 GEMM: a @ b^T * (sa * sb); out_dtype 0=bf16, "
          "1=fp8 e4m3 (requires out_scale)");
    m.def("linear_forward_fp8", &linear_forward_fp8, py::arg("x"),
          py::arg("w"), py::arg("bias"), py::arg("sx"), py::arg("sw"),
          py::arg("fmt") = 0,
          "Pure FP8 linear forward: quantize x/w, pre-quantized GEMM; "
          "returns (out, amax_x, amax_w)");
    m.def("linear_backward_fp8", &linear_backward_fp8, py::arg("g"),
          py::arg("x"), py::arg("w"), py::arg("masks"), py::arg("sg"),
          py::arg("sw"), py::arg("sx"), py::arg("fmt"),
          "FP8 linear backward; returns (grad_input, grad_weight, grad_bias, amax_g)");
}
