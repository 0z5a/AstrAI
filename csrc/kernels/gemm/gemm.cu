// GEMM family binding (module `gemm`): the pre-quantized FP8 matmul
// ``mm_fp8`` plus the family's launch entry. This TU is also the single
// kernel-policy instantiation site — the explicit instantiations below pin
// every production Policy to SASS exactly once; the C tests instantiate
// straight from the headers instead.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "common/device.cuh"
#include "gemm.cuh"
#include "quantize/common.h"

using namespace astrai;
using namespace astrai::quant;

namespace astrai {
namespace gemm {

// Explicit instantiation of the production entry points: every kernel
// Policy lands in SASS exactly once, here. Other TUs reach this through
// launch_gemm below (the header declares it); the C tests instantiate from
// the headers directly.
template void gemm<FP8Format::E4M3>(GemmParams, cudaStream_t, bool, bool);
template void gemm<FP8Format::E5M2>(GemmParams, cudaStream_t, bool, bool);

void launch_gemm(FP8Format fmt, const GemmParams& p, cudaStream_t stream,
                 bool trans_a, bool trans_b) {
    if (fmt == FP8Format::E5M2)
        gemm<FP8Format::E5M2>(p, stream, trans_a, trans_b);
    else
        gemm<FP8Format::E4M3>(p, stream, trans_a, trans_b);
}

namespace {

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
    GemmParams p;
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
    launch_gemm(a.scalar_type() == torch::kFloat8_e4m3fn ? FP8Format::E4M3
                                                         : FP8Format::E5M2,
                p, stream.stream(), tag_a, tag_b);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

}  // namespace

}  // namespace gemm
}  // namespace astrai

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm_fp8", &astrai::gemm::mm_fp8, py::arg("a"), py::arg("b"), py::arg("scale"),
          py::arg("trans_a") = false, py::arg("trans_b") = false,
          py::arg("bias") = py::none());
}
