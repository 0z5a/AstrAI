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

using namespace astrai::fp8;

namespace {

// FP8Format / FP8Params and the launchers live in astrai::fp8 (common.h /
// gemm.cuh); this TU opens the using-directive above so the binding reads
// them unqualified.

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
                      int64_t k, int64_t a_ld, int64_t b_ld) {
    p.a_ptr = a;
    p.b_ptr = b;
    p.out_ptr = out;
    p.scale_a = sa.data_ptr<float>();
    p.scale_b = sb.data_ptr<float>();
    p.out_scale = out_scale ? out_scale->data_ptr<float>() : nullptr;
    p.bias = nullptr;
    p.amax_a = nullptr;
    p.amax_b = nullptr;
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.a_ld = static_cast<int>(a_ld);
    p.b_ld = static_cast<int>(b_ld);
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
    p.a_ld = p.b_ld = 0;
    p.total = static_cast<int>(total);
}

// ---- GEMM launch dispatch (runtime flags -> compile-time kernel variants) ----

template <FP8Format Fmt, int Variant>
void launch_gemm_variant(const FP8Params& p, cudaStream_t stream) {
    static_assert(Variant >= 0 && Variant < 8,
                  "invalid FP8 GEMM dispatch variant");
    constexpr bool out_fp8 = (Variant & 4) != 0;
    // Variant bits 1/0 = trans_a/trans_b -> CUTLASS-style layout tags
    // (trans_a ? A ColMajor : RowMajor, same for B; see common.h).
    using LayoutA = std::conditional_t<(Variant & 2) != 0, ColMajor, RowMajor>;
    using LayoutB = std::conditional_t<(Variant & 1) != 0, ColMajor, RowMajor>;
    launch_fp8_gemm<Fmt, out_fp8, LayoutA, LayoutB>(p, stream);
}

template <FP8Format Fmt>
void dispatch_gemm(const FP8Params& p, cudaStream_t stream, bool out_fp8,
                   bool trans_a, bool trans_b) {
    // Encode the runtime flags as [output FP8, transpose A, transpose B].
    const int variant = (static_cast<int>(out_fp8) << 2) |
                        (static_cast<int>(trans_a) << 1) |
                        static_cast<int>(trans_b);
    switch (variant) {
        case 0: launch_gemm_variant<Fmt, 0>(p, stream); break;
        case 1: launch_gemm_variant<Fmt, 1>(p, stream); break;
        case 2: launch_gemm_variant<Fmt, 2>(p, stream); break;
        case 3: launch_gemm_variant<Fmt, 3>(p, stream); break;
        case 4: launch_gemm_variant<Fmt, 4>(p, stream); break;
        case 5: launch_gemm_variant<Fmt, 5>(p, stream); break;
        case 6: launch_gemm_variant<Fmt, 6>(p, stream); break;
        case 7: launch_gemm_variant<Fmt, 7>(p, stream); break;
    }
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
        launch_fp8_quantize<FP8Format::E5M2>(p, stream.stream());
    } else {
        launch_fp8_quantize<FP8Format::E4M3>(p, stream.stream());
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return {x8, amax};
}

torch::Tensor mm_fp8(torch::Tensor a, torch::Tensor b, torch::Tensor sa,
                     torch::Tensor sb, int64_t out_dtype,
                     c10::optional<torch::Tensor> out_scale, int64_t trans_a,
                     int64_t trans_b) {
    // Pre-quantized FP8 GEMM: out = op(a) @ op(b)^T * (sa * sb), FP32 accum.
    // trans_a / trans_b select the operand layout (0 = stored [M,K]/[K,N],
    // 1 = transposed [K,M]/[N,K]); the default (0/0) is the plain a @ b.
    // out_dtype: 0 = BF16 (default), 1 = FP8 E4M3 (requires out_scale, the
    // output quantization step). Both operands share one format.
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn ||
                    a.scalar_type() == torch::kFloat8_e5m2,
                "a and b must be fp8 (e4m3fn or e5m2)");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                "a and b must share the same fp8 format");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    check_scale(sa, a, "sa");
    check_scale(sb, a, "sb");
    check_fp8_device(a);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    const bool ta = (trans_a == 1), tb = (trans_b == 1);
    // Physical leading dimension = column count of each contiguous buffer.
    const int64_t a_ld = a_c.size(1);
    const int64_t b_ld = b_c.size(1);
    // Logical GEMM shape derived from the layout flags.
    const int64_t m = ta ? a_c.size(1) : a_c.size(0);
    const int64_t k = ta ? a_c.size(0) : a_c.size(1);
    const int64_t n = tb ? b_c.size(0) : b_c.size(1);
    const int64_t k2 = tb ? b_c.size(1) : b_c.size(0);
    TORCH_CHECK(k == k2, "inner dim mismatch");
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
                     out_fp8 ? &os : nullptr, m, n, k, a_ld, b_ld);
    if (a.scalar_type() == torch::kFloat8_e4m3fn)
        dispatch_gemm<FP8Format::E4M3>(p, stream.stream(), out_fp8, ta, tb);
    else
        dispatch_gemm<FP8Format::E5M2>(p, stream.stream(), out_fp8, ta, tb);
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
            launch_fp8_quantize<FP8Format::E5M2>(qp, stream.stream());
        } else {
            launch_fp8_quantize<FP8Format::E4M3>(qp, stream.stream());
        }
    };
    quantize(x_c, x8, sx, &amax_x);
    quantize(w_c, w8, sw, &amax_w);

    FP8Params p;
    // Forward is the NT layout: A = x8 [M,K] (a_ld = k), B = w8 [N,K]
    // (b_ld = k), out = x @ w^T. No operand transposes needed.
    pack_gemm_params(p, x8.data_ptr(), w8.data_ptr(), out.data_ptr(), sx, sw,
                     nullptr, m, n, k, k, k);
    if (fmt) {
        launch_fp8_gemm<FP8Format::E5M2, false, RowMajor, ColMajor>(
            p, stream.stream());
    } else {
        launch_fp8_gemm<FP8Format::E4M3, false, RowMajor, ColMajor>(
            p, stream.stream());
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
            launch_fp8_quantize<FP8Format::E5M2>(qp, stream.stream());
        } else {
            launch_fp8_quantize<FP8Format::E4M3>(qp, stream.stream());
        }
    };
    // Four-layout backward: the gradient and activation tensors keep their
    // natural row-major layout, and the kernel reads them transposed where the
    // GEMM needs it (the ColMajor layout tags pick the crosswise stage-load).
    // No torch-level `.transpose().contiguous()`
    // copies are required — dX uses g8 [M,N] as A with w8 [N,K] read transposed
    // as B; dW uses g8 transposed as A with x8 transposed as B.
    // g is quantized once (amax_g measured here); both GEMMs share g8.
    auto run_bwd_gemm = [&](const FP8Params& gp, bool trans_a, bool trans_b) {
        if (fmt)
            dispatch_gemm<FP8Format::E5M2>(gp, stream.stream(), false, trans_a,
                                           trans_b);
        else
            dispatch_gemm<FP8Format::E4M3>(gp, stream.stream(), false, trans_a,
                                           trans_b);
    };

    torch::Tensor g8;
    if (masks[0] || masks[1]) {
        g8 = torch::empty({m, n}, f8opt);
        quantize(g_c, g8, sg, &amax_g);
    }
    // dX = g @ w: A = g8 [M,N] (contract over N), B = w8 [N,K] read transposed
    // (b[p*b_ld + n] = w[p,n]); out = [M,K], a_ld = N, b_ld = K, contract = N.
    if (masks[0]) {
        auto w8 = torch::empty({n, k}, f8opt);
        quantize(w_c, w8, sw, nullptr);
        auto grad_input_2d = grad_input.reshape({m, k});
        FP8Params gp;
        pack_gemm_params(gp, g8.data_ptr(), w8.data_ptr(),
                         grad_input_2d.data_ptr(), sg, sw, nullptr, m, k, n, n,
                         k);
        run_bwd_gemm(gp, false, false);
    }
    // dW = g^T @ x: A = g8 [M,N] read transposed (a[p*a_ld + m] = g[p,m]), B =
    // x8 [M,K] read transposed (b[p*b_ld + n] = x[p,n]); out = [N,K], a_ld = N,
    // b_ld = K, contract = M.
    if (masks[1]) {
        auto x8 = torch::empty({m, k}, f8opt);
        quantize(x_c, x8, sx, nullptr);
        FP8Params gp;
        pack_gemm_params(gp, g8.data_ptr(), x8.data_ptr(),
                         grad_weight.data_ptr(), sg, sx, nullptr, n, k, m, n, k);
        run_bwd_gemm(gp, true, false);
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
          py::arg("out_scale") = py::none(), py::arg("trans_a") = 0,
          py::arg("trans_b") = 0,
          "Pre-quantized FP8 GEMM: op(a) @ op(b)^T * (sa * sb); out_dtype "
          "0=bf16, 1=fp8 e4m3 (requires out_scale); trans_a/trans_b select "
          "the operand layout (default 0/0 = a@b)");
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
