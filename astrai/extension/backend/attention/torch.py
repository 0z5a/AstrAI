"""Torch reference attention backend."""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from . import ATTN_BACKEND, AttentionBackend, AttentionBackendFactory, repeat_kv

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


@AttentionBackendFactory.register(ATTN_BACKEND.TORCH_NATIVE.value)
class TorchNativeBackend(AttentionBackend):
    """Reference backend using torch SDPA with indirect KV cache indexing.

    Writes new K/V into the cache buffers, gathers the full sequence K/V
    via ``req_to_token`` indirect indexing, then calls
    ``F.scaled_dot_product_attention``.

    Packed inference (3-D q) pads the ragged batch to [B, max_q, max_kv]
    and runs a single batched SDPA call with a combined causal+padding
    mask, then unpacks back to the flat layout.

    For training (``kv_cache is None``), skips cache I/O entirely and
    runs SDPA directly on the projected q/k/v.
    """

    priority = 100

    @classmethod
    def available(cls) -> bool:
        return True

    def supports_call(
        self,
        q: Tensor,
        kv_cache: Optional["KVCache"],
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
    ) -> bool:
        return True

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        fwd: Optional[str] = None,
    ) -> Tensor:
        self._check_fwd(fwd)
        if q.ndim == 4:
            n_rep = q.size(2) // k.size(2)
            if n_rep > 1:
                k = repeat_kv(k, n_rep)
                v = repeat_kv(v, n_rep)
            return (
                F.scaled_dot_product_attention(
                    q.permute(0, 2, 1, 3),
                    k.permute(0, 2, 1, 3),
                    v.permute(0, 2, 1, 3),
                    attn_mask,
                    is_causal=is_causal,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        if kv_cache is None or kv_cache.qo_indptr is None:
            raise ValueError("packed attention requires KV cache metadata")
        kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
        kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v

        # Pad the ragged batch to [B, max_q, max_kv] so one batched SDPA call
        # replaces B per-request calls. The bool mask folds the per-request
        # causal offset (seq_len - q_len) and the kv padding together; padded
        # q rows gather slot/row 0 and are dropped by the [q_valid] unpack,
        # which restores the packed qo_indptr order.
        qo_indptr = kv_cache.qo_indptr
        q_lens = qo_indptr[1:] - qo_indptr[:-1]
        seq_lens = kv_cache.seq_lens
        max_q = int(q_lens.max())
        max_kv = int(seq_lens.max())

        pos = torch.arange(max_kv, device=q.device)
        kv_valid = pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        token_index = kv_cache.req_to_token[kv_cache.req_pool_indices, :max_kv]
        token_index = token_index.masked_fill(~kv_valid, 0)
        k_b = kv_cache.k_buffer[layer_id, token_index]
        v_b = kv_cache.v_buffer[layer_id, token_index]
        n_rep = q.size(1) // k.size(1)
        if n_rep > 1:
            k_b = repeat_kv(k_b, n_rep)
            v_b = repeat_kv(v_b, n_rep)

        q_pos = torch.arange(max_q, device=q.device)
        q_valid = q_pos.unsqueeze(0) < q_lens.unsqueeze(1)
        q_index = (qo_indptr[:-1].unsqueeze(1) + q_pos.unsqueeze(0)).masked_fill(
            ~q_valid, 0
        )
        causal = (
            (seq_lens - q_lens).unsqueeze(1).unsqueeze(2) + q_pos.view(1, max_q, 1)
        ) >= pos.view(1, 1, max_kv)
        mask = (causal & kv_valid.unsqueeze(1)).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q[q_index].permute(0, 2, 1, 3),
            k_b.permute(0, 2, 1, 3),
            v_b.permute(0, 2, 1, 3),
            attn_mask=mask,
        )
        return out.permute(0, 2, 1, 3)[q_valid]
