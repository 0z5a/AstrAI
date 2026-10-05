/*
 * SGLang-style paged GQA decode (flat KV pool + req_to_token + kv_indptr)
 * — the implementation of the entry declared in api/attention.h.
 * Device-side code is in kernel/attention/launch.cuh +
 * kernel/attention/split_kv.cuh.
 */

#include "entry.h"
#include <api/attention.h>
#include <api/attention_dtypes.h>
#include <kernel/attention/launch.cuh>

namespace astrai {
namespace attention {

torch::Tensor attn_paged_decode(torch::Tensor q,
                                torch::Tensor k_cache,
                                torch::Tensor v_cache,
                                torch::Tensor req_to_token,
                                torch::Tensor req_pool_indices,
                                torch::Tensor kv_indptr,
                                c10::optional<torch::Tensor> new_k,
                                c10::optional<torch::Tensor> new_v,
                                c10::optional<torch::Tensor> mask,
                                int64_t causal_offset,
                                double scale,
                                c10::optional<torch::Tensor> o_part_buf,
                                c10::optional<torch::Tensor> ml_part_buf,
                                c10::optional<torch::Tensor> out_buf) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams p;
    attn_pack_paged_decode_params(q, k_cache, v_cache, req_to_token, req_pool_indices, kv_indptr,
                                  new_k, new_v, mask, causal_offset, scale, p);

    torch::Tensor O;
    if (out_buf.has_value() && out_buf->defined()) {
        TORCH_CHECK(out_buf->dtype() == q.dtype(), "out_buf dtype must match q");
        TORCH_CHECK(out_buf->is_cuda() && out_buf->is_contiguous(),
                    "out_buf must be a contiguous CUDA tensor");
        TORCH_CHECK(out_buf->size(0) >= q.size(0), "out_buf batch too small");
        TORCH_CHECK(out_buf->size(1) == q.size(1), "out_buf heads must match q");
        TORCH_CHECK(out_buf->size(2) == q.size(2), "out_buf head_dim must match q");
        TORCH_CHECK(q.is_contiguous(), "q must be contiguous when out_buf is provided");
        O = out_buf.value().slice(0, 0, q.size(0));
    } else {
        O = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options());
    }
    p.o_ptr = O.data_ptr();

    resolve_split_buffers(o_part_buf, ml_part_buf, p);
    attn_dtype_dispatch<AttnDispatchPagedDecode>(q.scalar_type(), p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

} // namespace attention
} // namespace astrai
