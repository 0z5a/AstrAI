#include "attn_dispatchers.cuh"
#include "attn_entry_utils.cuh"

torch::Tensor attn_decode(
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
    TORCH_CHECK(p.q_len == 1, "Q seq_len must be 1");
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == BLHD) ? O.transpose(1, 2) : O;
    p.o = (bf16*)O_view.data_ptr();

    {
        static torch::Tensor s_o_part, s_ml_part;
        int64_t o_needed = (int64_t)p.batch * p.q_head * MAX_SPLITS * p.head_dim;
        auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        if (!s_o_part.defined() || s_o_part.numel() < o_needed) {
            s_o_part = torch::empty({p.batch, p.q_head, MAX_SPLITS, p.head_dim}, fopt);
            s_ml_part = torch::empty({p.batch, p.q_head, MAX_SPLITS, 2}, fopt);
        }
        p.o_part = (float*)s_o_part.data_ptr();
        p.ml_part = (float*)s_ml_part.data_ptr();
    }
    DISPATCH_HEAD_DIM(p.head_dim, dispatch_decode, p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_decode", &attn_decode,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("layout") = (int64_t)BHLD,
        "GQA decode (tensor-core head-packing on sm_80+, scalar fallback)");
}
