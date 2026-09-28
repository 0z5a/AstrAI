// GQA decode (split-KV FlashDecoding), contiguous K/V — the implementation
// of the entry declared in api/attention.h (the gated_deltanet_fwd.cu
// shape). Device-side code (kernels, launchers, dispatchers) is in
// kernel/attention_launch.cuh + kernel/attention_split_kv.cuh.

#include "entry.h"
#include <api/attention.h>
#include <api/attention_dtypes.h>
#include <kernel/attention_launch.cuh>

namespace astrai {
namespace attention {

torch::Tensor attn_decode(torch::Tensor q,
                          torch::Tensor k,
                          torch::Tensor v,
                          c10::optional<torch::Tensor> mask,
                          int64_t causal_offset,
                          double scale,
                          int64_t layout,
                          c10::optional<torch::Tensor> o_part_buf,
                          c10::optional<torch::Tensor> ml_part_buf) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams p;
    attn_pack_params(q, k, v, mask, causal_offset, scale, layout, p);
    TORCH_CHECK(p.q_len == 1, "Q seq_len must be 1");
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == BLHD) ? O.transpose(1, 2) : O;
    p.o_ptr = O_view.data_ptr();

    resolve_split_buffers(o_part_buf, ml_part_buf, p);
    attn_dtype_dispatch<AttnDispatchDecode>(q.scalar_type(), p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

} // namespace attention
} // namespace astrai
