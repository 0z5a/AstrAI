#include "dispatchers.cuh"
#include "entry_utils.cuh"

using namespace astrai::attention;

torch::Tensor attn_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams<bf16> p;
    attn_pack_params(q, k, v, mask, causal_offset, scale, layout, p);
    TORCH_CHECK(p.head_dim % 16 == 0, "head_dim must be multiple of 16");

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == BLHD) ? O.transpose(1, 2) : O;
    p.o_ptr = (bf16*)O_view.data_ptr();

    DISPATCH_HEAD_DIM(p.head_dim, dispatch_prefill, p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_prefill", &attn_prefill,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("layout") = (int64_t)BHLD,
        "GQA prefill (tensor-core mma on sm_80+, scalar fallback)");
}
