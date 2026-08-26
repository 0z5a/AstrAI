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

// Inner-layout resolution for one GEMM operand. The user flag names the
// math (0 = tensor's last two dims are [rows][contract], 1 = transposed);
// the storage may independently be a col-major view (.t() of a contiguous
// buffer), which folds into the returned dispatch flag at zero copy — the
// kernel's LayoutA/LayoutB tags cover both storages. m/n/k derive from the
// user flag only; the fold never swaps them (see the layout table in
// gemm.cuh). Tensors whose inner dims are neither natural layout fall back
// to .contiguous().
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
                     int64_t trans_a, int64_t trans_b, torch::Tensor bias) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn ||
                    a.scalar_type() == torch::kFloat8_e5m2,
                "a and b must be fp8");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must share format");
    TORCH_CHECK((a.dim() == 2 || a.dim() == 3) &&
                    (b.dim() == 2 || b.dim() == 3),
                "a and b must be 2D or 3D (batched)");
    TORCH_CHECK(a.device() == b.device(), "a and b must share device");
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
    const bool tag_a =
        resolve_operand(a, trans_a != 0, a_ld, a_bstride, a_st);
    const bool tag_b =
        resolve_operand(b, trans_b != 0, b_ld, b_bstride, b_st);
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
    pack_gemm(p, a_st.data_ptr(), b_st.data_ptr(), output.data_ptr(), scale,
              m, n, k, a_ld, b_ld);
    // Fused epilogue bias (bf16, broadcast over rows and batches). An
    // undefined or 0-element tensor keeps the plain scaled output.
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kBFloat16,
                    "fp8 gemm bias must be a CUDA bf16 tensor");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == n,
                    "fp8 gemm bias must be 1D of length n=", n);
        TORCH_CHECK(bias.is_contiguous(), "fp8 gemm bias must be contiguous");
        p.bias_ptr = bias.data_ptr();
    }
    p.batch = static_cast<int>(batch);
    p.a_batch_stride = (batch_a == 1 && batch > 1) ? 0 : a_bstride;
    p.b_batch_stride = (batch_b == 1 && batch > 1) ? 0 : b_bstride;
    p.out_batch_stride = m * n;
    if (a.scalar_type() == torch::kFloat8_e4m3fn)
        dispatch_gemm<FP8Format::E4M3>(p, stream.stream(), tag_a, tag_b);
    else
        dispatch_gemm<FP8Format::E5M2>(p, stream.stream(), tag_a, tag_b);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

// mm_fp8 binding: Python None and an omitted argument both mean "no bias"
// (resolved to an undefined tensor here, so every Python layer can pass its
// bias argument through untouched instead of normalizing it host-side).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize, py::arg("x"), py::arg("scale"),
          py::arg("fmt"));
    m.def(
        "mm_fp8",
        [](torch::Tensor a, torch::Tensor b, torch::Tensor scale,
           int64_t trans_a, int64_t trans_b, py::object bias) {
            torch::Tensor t;
            if (!bias.is_none()) {
                // (py::isinstance<torch::Tensor> is false for real tensors
                // here — torch's caster registers no pybind type info — so
                // validate by attempting the cast itself.)
                try {
                    t = bias.cast<torch::Tensor>();
                } catch (const py::cast_error&) {
                    TORCH_CHECK(false, "bias must be a torch.Tensor or None");
                }
            }
            return mm_fp8(a, b, scale, trans_a, trans_b, t);
        },
        py::arg("a"), py::arg("b"), py::arg("scale"), py::arg("trans_a") = 0,
        py::arg("trans_b") = 0, py::arg("bias") = py::none());
}
