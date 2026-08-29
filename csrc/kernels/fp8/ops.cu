// CUDA bindings for the stateless FP8 quantize/GEMM primitives.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <mutex>
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

// Inner-layout resolution for one GEMM operand. The user flag names the
// math (0 = last two dims are [rows][contract], 1 = transposed); the
// storage may independently be a col-major view (.t() of a contiguous
// buffer), which folds into the returned dispatch flag at zero copy — the
// kernel's LayoutA/LayoutB tags cover both storages. m/n/k derive from the
// user flag only. Tensors whose inner dims are neither natural layout fall
// back to .contiguous().
bool resolve_operand(const torch::Tensor& t_in, bool flag, int64_t& ld,
                     int64_t& batch_stride, torch::Tensor& storage) {
    torch::Tensor t = t_in;
    bool col_major = false;
    if (t.stride(-1) != 1) {
        if (t.stride(-2) == 1) {
            col_major = true;
        } else {
            t = t.contiguous();
        }
    }
    storage = t;
    ld = col_major ? t.stride(-1) : t.stride(-2);
    batch_stride = t.dim() == 3 ? t.stride(0) : 0;
    return flag ^ col_major;
}

// Dtype dispatch over the unified quantize launcher.
template <bool Tiled, FP8Format Fmt>
void launch_for_dtype(const torch::Tensor& x, const FP8QuantizeParams& p,
                      cudaStream_t stream) {
    switch (x.scalar_type()) {
    case torch::kHalf:
        launch_fp8_quantize<Fmt, __half, Tiled>(p, stream);
        break;
    case torch::kFloat32:
        launch_fp8_quantize<Fmt, float, Tiled>(p, stream);
        break;
    default:
        launch_fp8_quantize<Fmt, __nv_bfloat16, Tiled>(p, stream);
    }
}

template <bool Tiled>
void launch_quantize_for(const torch::Tensor& x, const FP8QuantizeParams& p,
                         bool e5m2, cudaStream_t stream) {
    if (e5m2)
        launch_for_dtype<Tiled, FP8Format::E5M2>(x, p, stream);
    else
        launch_for_dtype<Tiled, FP8Format::E4M3>(x, p, stream);
}

// Shared binding body for the two quantize entry points: RowMajor /
// Transposed (single output) serve quantize(), Dual (both orientations from
// one read) serves quantize_dual(). A ring tensor switches
// on the in-kernel delayed-scaling fold: state layout
// [hist n | scale | legacy | amax | done-as-int], and the returned amax is
// the (self-cleaned) persistent slot. Without it, amax is reduced into a
// fresh buffer armed by a driver memset — cheaper than the zeros() fill
// kernel.
py::object quantize_impl(torch::Tensor x, torch::Tensor scale, int64_t fmt,
                         QuantLayout layout, py::object ring, int64_t hist_idx,
                         double fp8_max, double pow2_margin) {
    TORCH_CHECK(x.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 ||
                    x.scalar_type() == torch::kHalf ||
                    x.scalar_type() == torch::kFloat32,
                "x must be bf16, fp16 or fp32");
    TORCH_CHECK(fmt == static_cast<int64_t>(FP8Format::E4M3) ||
                    fmt == static_cast<int64_t>(FP8Format::E5M2),
                "unsupported quantization type: expected E4M3 (0) or E5M2 (1)");
    TORCH_CHECK(layout == QuantLayout::RowMajor || x.dim() >= 2,
                "transposed quantize layouts need a 2D+ tensor");
    check_scale(scale, x);
    check_fp8_device(x);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto input = x.contiguous();
    auto out_opts = input.options().dtype(
        fmt ? torch::kFloat8_e5m2 : torch::kFloat8_e4m3fn);
    torch::Tensor amax;
    float *ring_hist = nullptr, *ring_scale_out = nullptr;
    unsigned int* ring_done = nullptr;
    int ring_len = 0;
    if (!ring.is_none()) {
        auto st = ring.cast<torch::Tensor>();
        TORCH_CHECK(st.is_cuda() && st.dim() == 1 &&
                        st.scalar_type() == torch::kFloat32,
                    "ring state must be a 1D float32 CUDA tensor");
        const int64_t n = st.numel() - 4;
        TORCH_CHECK(n > 0 && hist_idx >= 0 && hist_idx < n,
                    "ring state too small or hist_idx out of range");
        float* base = st.data_ptr<float>();
        amax = st.narrow(0, n + 2, 1);
        ring_hist = base;
        ring_scale_out = base + n;
        ring_done = reinterpret_cast<unsigned int*>(base + n + 3);
        ring_len = static_cast<int>(n);
    } else {
        amax = torch::empty({1}, input.options().dtype(torch::kFloat32));
        cudaMemsetAsync(amax.data_ptr(), 0, sizeof(float), stream.stream());
    }

    FP8QuantizeParams p;
    p.input_ptr = input.data_ptr();
    p.scale = scale.data_ptr<float>();
    p.amax = amax.data_ptr<float>();
    if (ring_hist) {
        p.fold_ring = true;
        p.hist = ring_hist;
        p.scale_out = ring_scale_out;
        p.done = ring_done;
        p.hist_len = ring_len;
        p.hist_idx = static_cast<int>(hist_idx);
        p.fp8_max = static_cast<float>(fp8_max);
        p.pow2_margin = static_cast<float>(pow2_margin);
    }
    p.total = static_cast<int>(input.numel());
    p.out_layout = layout;
    p.rows = static_cast<int>(input.size(-2));
    p.cols = static_cast<int>(input.size(-1));
    torch::Tensor output, output_t;
    if (layout != QuantLayout::Transposed) {
        output = torch::empty_like(input, out_opts);
        p.output_ptr = output.data_ptr();
    }
    if (layout != QuantLayout::RowMajor) {
        output_t = torch::empty({input.size(-1), input.size(-2)}, out_opts);
        p.output_transposed_ptr = output_t.data_ptr();
    }
    const bool e5m2 = fmt == static_cast<int64_t>(FP8Format::E5M2);
    if (layout == QuantLayout::RowMajor)
        launch_quantize_for<false>(input, p, e5m2, stream.stream());
    else
        launch_quantize_for<true>(input, p, e5m2, stream.stream());
    C10_CUDA_CHECK(cudaGetLastError());
    if (layout == QuantLayout::Dual)
        return py::make_tuple(output, output_t, amax);
    return py::make_tuple(
        layout == QuantLayout::Transposed ? output_t : output, amax);
}

}  // namespace

// Single-orientation quantize binding: row-major x8, or its [cols][rows]
// transpose when transposed is set — the K-contiguous operand orientation
// NT GEMMs want. Returns (x8|x8T, amax).
py::object quantize(torch::Tensor x, torch::Tensor scale, int64_t fmt,
                    bool transposed, py::object ring, int64_t hist_idx,
                    double fp8_max, double pow2_margin) {
    const QuantLayout layout =
        transposed ? QuantLayout::Transposed : QuantLayout::RowMajor;
    return quantize_impl(x, scale, fmt, layout, ring, hist_idx, fp8_max,
                         pow2_margin);
}

// Dual-orientation quantize binding: one read of x produces both the
// row-major x8 and its transpose (plus amax), for tensors consumed by GEMMs
// in both orientations (backward g). Returns (x8, x8T, amax).
py::object quantize_dual(torch::Tensor x, torch::Tensor scale, int64_t fmt,
                         py::object ring, int64_t hist_idx, double fp8_max,
                         double pow2_margin) {
    return quantize_impl(x, scale, fmt, QuantLayout::Dual, ring, hist_idx,
                         fp8_max, pow2_margin);
}

torch::Tensor mm_fp8(torch::Tensor a, torch::Tensor b, torch::Tensor scale,
                     bool trans_a, bool trans_b, py::object bias) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn ||
                    a.scalar_type() == torch::kFloat8_e5m2,
                "a and b must be fp8");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must share format");
    TORCH_CHECK((a.dim() == 2 || a.dim() == 3) &&
                    (b.dim() == 2 || b.dim() == 3),
                "a and b must be 2D or 3D (batched)");
    TORCH_CHECK(a.device() == b.device(), "a and b must share device");
    // Python None and an omitted argument both mean "no bias" — an undefined
    // tensor below. (py::isinstance<torch::Tensor> is false for real tensors
    // here — torch's caster registers no pybind type info — so validate by
    // attempting the cast itself.)
    torch::Tensor bias_t;
    if (!bias.is_none()) {
        try {
            bias_t = bias.cast<torch::Tensor>();
        } catch (const py::cast_error&) {
            TORCH_CHECK(false, "bias must be a torch.Tensor or None");
        }
    }
    check_scale(scale, a);
    check_fp8_device(a);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Batched operands follow matmul broadcast rules: 2D acts as a batch
    // of 1; a size-1 batch broadcasts across the other side (stride 0).
    const int64_t batch_a = a.dim() == 3 ? a.size(0) : 1;
    const int64_t batch_b = b.dim() == 3 ? b.size(0) : 1;
    TORCH_CHECK(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                "batch dim mismatch (got ", batch_a, " and ", batch_b, ")");
    const int64_t batch = std::max(batch_a, batch_b);
    TORCH_CHECK(batch <= 65535, "batch dim exceeds the grid.z launch limit");

    torch::Tensor a_st, b_st;
    int64_t a_ld, b_ld, a_bstride, b_bstride;
    const bool tag_a = resolve_operand(a, trans_a, a_ld, a_bstride, a_st);
    const bool tag_b = resolve_operand(b, trans_b, b_ld, b_bstride, b_st);
    // GEMM dims from the user flags; storage layout never swaps them.
    const int64_t m = trans_a ? a.size(-1) : a.size(-2);
    const int64_t k = trans_a ? a.size(-2) : a.size(-1);
    const int64_t n = trans_b ? b.size(-2) : b.size(-1);
    TORCH_CHECK(k == (trans_b ? b.size(-1) : b.size(-2)), "inner dim mismatch");

    const bool batched_out = a.dim() == 3 || b.dim() == 3;
    torch::Tensor output =
        batched_out
            ? torch::empty({batch, m, n}, a.options().dtype(torch::kBFloat16))
            : torch::empty({m, n}, a.options().dtype(torch::kBFloat16));
    FP8Params p;
    p.a_ptr = a_st.data_ptr();
    p.b_ptr = b_st.data_ptr();
    p.out_ptr = output.data_ptr();
    p.scale = scale.data_ptr<float>();
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.a_ld = static_cast<int>(a_ld);
    p.b_ld = static_cast<int>(b_ld);
    // Fused epilogue bias (bf16, broadcast over rows and batches). An
    // undefined or 0-element tensor keeps the plain scaled output.
    if (bias_t.defined() && bias_t.numel() > 0) {
        TORCH_CHECK(bias_t.is_cuda() && bias_t.scalar_type() == torch::kBFloat16,
                    "fp8 gemm bias must be a CUDA bf16 tensor");
        TORCH_CHECK(bias_t.dim() == 1 && bias_t.size(0) == n,
                    "fp8 gemm bias must be 1D of length n=", n);
        TORCH_CHECK(bias_t.is_contiguous(), "fp8 gemm bias must be contiguous");
        p.bias_ptr = bias_t.data_ptr();
    }
    p.batch = static_cast<int>(batch);
    p.a_batch_stride = (batch_a == 1 && batch > 1) ? 0 : a_bstride;
    p.b_batch_stride = (batch_b == 1 && batch > 1) ? 0 : b_bstride;
    p.out_batch_stride = m * n;
    if (a.scalar_type() == torch::kFloat8_e4m3fn)
        gemm<FP8Format::E4M3>(p, stream.stream(), tag_a, tag_b);
    else
        gemm<FP8Format::E5M2>(p, stream.stream(), tag_a, tag_b);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize, py::arg("x"), py::arg("scale"),
          py::arg("fmt"), py::arg("transposed") = false,
          py::arg("ring") = py::none(), py::arg("hist_idx") = 0,
          py::arg("fp8_max") = 448.0, py::arg("pow2_margin") = 1.0);
    m.def("quantize_dual", &quantize_dual, py::arg("x"), py::arg("scale"),
          py::arg("fmt"), py::arg("ring") = py::none(),
          py::arg("hist_idx") = 0, py::arg("fp8_max") = 448.0,
          py::arg("pow2_margin") = 1.0);
    m.def("mm_fp8", &mm_fp8, py::arg("a"), py::arg("b"), py::arg("scale"),
          py::arg("trans_a") = false, py::arg("trans_b") = false,
          py::arg("bias") = py::none());
}
