#include "dispatchers.cuh"
#include "entry_utils.cuh"

torch::Tensor attn_decode(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout,
    c10::optional<torch::Tensor> o_part_buf,
    c10::optional<torch::Tensor> ml_part_buf
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams<bf16> p;
    attn_pack_params(q, k, v, mask, causal_offset, scale, layout, p);
    TORCH_CHECK(p.q_len == 1, "Q seq_len must be 1");
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == BLHD) ? O.transpose(1, 2) : O;
    p.o_ptr = (bf16*)O_view.data_ptr();

    if (o_part_buf.has_value() && ml_part_buf.has_value()
        && o_part_buf->defined() && ml_part_buf->defined()) {
        TORCH_CHECK(o_part_buf->scalar_type() == torch::kFloat32, "o_part_buf must be f32");
        TORCH_CHECK(ml_part_buf->scalar_type() == torch::kFloat32, "ml_part_buf must be f32");
        int64_t o_needed = (int64_t)p.batch * p.q_head * MAX_SPLITS * p.head_dim;
        int64_t ml_needed = (int64_t)p.batch * p.q_head * MAX_SPLITS * 2;
        TORCH_CHECK(o_part_buf->numel() >= o_needed,
                     "o_part_buf too small: need ", o_needed, " got ", o_part_buf->numel());
        TORCH_CHECK(ml_part_buf->numel() >= ml_needed,
                     "ml_part_buf too small: need ", ml_needed, " got ", ml_part_buf->numel());
        TORCH_CHECK(o_part_buf->is_cuda() && ml_part_buf->is_cuda(),
                     "split buffers must be CUDA tensors");
        TORCH_CHECK(o_part_buf->is_contiguous() && ml_part_buf->is_contiguous(),
                     "split buffers must be contiguous");
        p.o_part = (float*)o_part_buf->data_ptr();
        p.ml_part = (float*)ml_part_buf->data_ptr();
    } else {
        alloc_split_partials(p);
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
        py::arg("o_part_buf") = py::none(),
        py::arg("ml_part_buf") = py::none(),
        "GQA decode (tensor-core head-packing on sm_80+, scalar fallback)");
}
