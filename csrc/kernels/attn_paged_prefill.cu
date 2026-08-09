#include "attn_dispatchers.cuh"
#include "attn_entry_utils.cuh"

torch::Tensor attn_paged_prefill(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor kv_indptr,
    torch::Tensor qo_indptr,
    c10::optional<torch::Tensor> mask,
    int64_t max_q_len,
    int64_t causal_offset,
    double scale
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams<bf16> p;
    attn_pack_paged_prefill_params(q, k_cache, v_cache,
                                    req_to_token, req_pool_indices,
                                    kv_indptr, qo_indptr, mask,
                                    max_q_len, causal_offset, scale, p);

    auto O = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options());
    p.o_ptr = (bf16*)O.data_ptr();

    DISPATCH_HEAD_DIM(p.head_dim, dispatch_paged_prefill, p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_paged_prefill", &attn_paged_prefill,
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("req_to_token"),
        py::arg("req_pool_indices"),
        py::arg("kv_indptr"),
        py::arg("qo_indptr"),
        py::arg("mask") = py::none(),
        py::arg("max_q_len"),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        "SGLang-style paged prefill: flat KV pool + ragged batch.");
}
