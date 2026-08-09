#include "attn_dispatchers.cuh"
#include "attn_entry_utils.cuh"

torch::Tensor attn_paged_decode(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor kv_indptr,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    c10::optional<torch::Tensor> o_part_buf,
    c10::optional<torch::Tensor> ml_part_buf,
    c10::optional<torch::Tensor> out_buf
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams<bf16> p;
    attn_pack_paged_decode_params(q, k_cache, v_cache,
                                   req_to_token, req_pool_indices, kv_indptr,
                                   mask, causal_offset, scale, p);

    torch::Tensor O;
    if (out_buf.has_value() && out_buf->defined()) {
        TORCH_CHECK(out_buf->dtype() == q.dtype(), "out_buf dtype must match q");
        TORCH_CHECK(out_buf->is_cuda() && out_buf->is_contiguous(),
                    "out_buf must be a contiguous CUDA tensor");
        TORCH_CHECK(out_buf->size(0) >= q.size(0), "out_buf batch too small");
        TORCH_CHECK(out_buf->size(1) == q.size(1), "out_buf heads must match q");
        TORCH_CHECK(out_buf->size(2) == q.size(2), "out_buf head_dim must match q");
        TORCH_CHECK(q.is_contiguous(),
                    "q must be contiguous when out_buf is provided");
        O = out_buf.value().slice(0, 0, q.size(0));
    } else {
        O = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options());
    }
    p.o_ptr = (bf16*)O.data_ptr();

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
    DISPATCH_HEAD_DIM(p.head_dim, dispatch_paged_decode, p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_paged_decode", &attn_paged_decode,
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("req_to_token"),
        py::arg("req_pool_indices"),
        py::arg("kv_indptr"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("o_part_buf") = py::none(),
        py::arg("ml_part_buf") = py::none(),
        py::arg("out_buf") = py::none(),
        "SGLang-style paged decode: flat KV pool + req_to_token + kv_indptr.");
}
