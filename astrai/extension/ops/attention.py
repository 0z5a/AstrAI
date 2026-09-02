"""Attention kernel wrapper functions - one entry point per compiled kernel.

Each wrapper calls its CUDA kernel directly. If the kernel is not
available, raises ``RuntimeError``. Fallback to torch SDPA is the
responsibility of the attention backend, not this module.

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). Scale is always ``1/sqrt(head_dim)``.

Interface (all functions):
    is_causal: True = causal mask; False = non-causal
    mask:      2D [batch, kv_len] or 3D [batch, q_len, kv_len] (bool, True=keep)
"""

import enum
from typing import Optional

import torch

from astrai.extension.loader import get_module


class TensorLayout(enum.IntEnum):
    """Q/K/V tensor layout, mirrors the C++ ``TensorLayout`` enum in ``attn_common.h``.

    Kernels internally operate on BHLD; BLHD inputs are transposed at entry.
    """

    BHLD = 0  # [batch, n_heads, seq_len, head_dim]
    BLHD = 1  # [batch, seq_len, n_heads, head_dim]


def attn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """GQA decode attention (q_len == 1).

    Args:
        q: [batch, 1, n_heads, head_dim] (blhd, bf16)
        k: [batch, kv_len, n_kv_heads, head_dim] (blhd, bf16)
        v: [batch, kv_len, n_kv_heads, head_dim] (blhd, bf16)
        mask: 2D [batch, kv_len] or 3D [batch, 1, kv_len] (bool, True=keep)
        is_causal: apply causal mask

    Returns:
        [batch, 1, n_heads, head_dim] (blhd, bf16)
    """
    mod = get_module("attn_decode")
    causal_offset = (k.size(1) - 1) if is_causal else -1
    return mod.attn_decode(
        q, k, v, mask=mask, causal_offset=causal_offset, layout=TensorLayout.BLHD
    )


def attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """GQA prefill attention (q_len > 1).

    Args:
        q: [batch, q_len, n_heads, head_dim] (blhd, bf16)
        k: [batch, kv_len, n_kv_heads, head_dim] (blhd, bf16)
        v: [batch, kv_len, n_kv_heads, head_dim] (blhd, bf16)
        mask: 2D [batch, kv_len] or 3D [batch, q_len, kv_len] (bool, True=keep)
        is_causal: apply causal mask

    Returns:
        [batch, q_len, n_heads, head_dim] (blhd, bf16)
    """
    mod = get_module("attn_prefill")
    causal_offset = (k.size(1) - q.size(1)) if is_causal else -1
    return mod.attn_prefill(
        q, k, v, mask=mask, causal_offset=causal_offset, layout=TensorLayout.BLHD
    )


def attn_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    new_k: Optional[torch.Tensor] = None,
    new_v: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    o_part_buf: Optional[torch.Tensor] = None,
    ml_part_buf: Optional[torch.Tensor] = None,
    out_buf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SGLang-style paged decode (q_len == 1, flat KV pool).

    Reads K/V directly from a flat pool [size, kv_head, head_dim] via
    req_to_token indirect indexing.  Each request has its own seq_len
    (from kv_indptr), eliminating padding waste.

    Args:
        q: [batch, n_heads, head_dim] (bf16, 3D — no seq dim)
        k_cache: [pool_size, n_kv_heads, head_dim] (bf16, flat)
        v_cache: same as k_cache
        req_to_token: [num_reqs, max_context_len] (int32) — token -> slot
        req_pool_indices: [batch] (int32) — rows into req_to_token
        kv_indptr: [batch+1] (int32) — prefix sum of per-request seq_lens
        new_k: current-token K to append, [batch, n_kv_heads, head_dim]
        new_v: current-token V to append, same shape as new_k
        mask: 2D [batch, max_context_len] (bool, True=keep) or None
        is_causal: apply causal mask
        o_part_buf: pre-allocated split-KV o partial buffer (workflow bypass)
        ml_part_buf: pre-allocated split-KV m/l buffer (workflow bypass)
        out_buf: pre-allocated output buffer [batch, n_heads, head_dim] (graph-safe)

    Returns:
        [batch, n_heads, head_dim] (bf16, 3D)
    """
    mod = get_module("attn_paged_decode")
    causal_offset = 0 if is_causal else -1
    return mod.attn_paged_decode(
        q,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        kv_indptr,
        new_k=new_k,
        new_v=new_v,
        mask=mask,
        causal_offset=causal_offset,
        o_part_buf=o_part_buf,
        ml_part_buf=ml_part_buf,
        out_buf=out_buf,
    )


def attn_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    qo_indptr: torch.Tensor,
    q_tile_to_batch: torch.Tensor,
    q_tile_to_index: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """SGLang-style paged prefill (ragged batch, flat KV pool).

    Reads K/V directly from a flat pool [size, kv_head, head_dim] via
    req_to_token.  Supports ragged batches: each request has its own
    q_len and kv_len, addressed via qo_indptr and kv_indptr.

    Args:
        q: [total_q, n_heads, head_dim] (bf16, 3D — flattened across requests)
        k_cache: [pool_size, n_kv_heads, head_dim] (bf16, flat)
        v_cache: same as k_cache
        req_to_token: [num_reqs, max_context_len] (int32)
        req_pool_indices: [batch] (int32)
        kv_indptr: [batch+1] (int32) — prefix sum of per-request kv_lens
        qo_indptr: [batch+1] (int32) — prefix sum of per-request q_lens
        q_tile_to_batch: [num_q_tiles] (int32) — request index per Q tile
        q_tile_to_index: [num_q_tiles] (int32) — local Q tile index per request
        mask: 4D [batch, 1, q_len, kv_len] (bool, True=keep) or None
        is_causal: apply causal mask

    Returns:
        [total_q, n_heads, head_dim] (bf16, 3D)
    """
    mod = get_module("attn_paged_prefill")
    causal_offset = 0 if is_causal else -1
    return mod.attn_paged_prefill(
        q,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        kv_indptr,
        qo_indptr,
        q_tile_to_batch,
        q_tile_to_index,
        mask,
        causal_offset=causal_offset,
    )


__all__ = [
    "TensorLayout",
    "attn_decode",
    "attn_paged_decode",
    "attn_paged_prefill",
    "attn_prefill",
]
