"""Attention kernel wrapper functions — one entry point per compiled kernel.

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

from astrai.extension.loader import _available, _modules


class TensorLayout(enum.IntEnum):
    """Q/K/V tensor layout, mirrors the C++ ``TensorLayout`` enum in ``attn_common.h``.

    Kernels internally operate on BHLD; BLHD inputs are transposed at entry.
    """

    BHLD = 0  # [batch, n_heads, seq_len, head_dim]
    BLHD = 1  # [batch, seq_len, n_heads, head_dim]


def _check_available(name: str):
    if not _available.get(name):
        raise RuntimeError(
            f"CUDA kernel '{name}' is not available. "
            f"Build with CSRC_KERNELS=true or use a torch-native backend."
        )


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
    _check_available("attn_decode")
    causal_offset = (k.size(1) - 1) if is_causal else -1
    return _modules["attn_decode"].attn_decode(
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
    _check_available("attn_prefill")
    causal_offset = (k.size(1) - q.size(1)) if is_causal else -1
    return _modules["attn_prefill"].attn_prefill(
        q, k, v, mask=mask, causal_offset=causal_offset, layout=TensorLayout.BLHD
    )


def attn_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    max_seq_len: int,
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
        req_to_token: [num_reqs, max_context_len] (int64) — token -> slot
        req_pool_indices: [batch] (int64) — rows into req_to_token
        kv_indptr: [batch+1] (int32) — prefix sum of per-request seq_lens
        max_seq_len: max per-request seq_len (Python int, for split computation)
        mask: 2D [batch, max_seq_len] (bool, True=keep) or None
        is_causal: apply causal mask
        o_part_buf: pre-allocated split-KV o partial buffer (workflow bypass)
        ml_part_buf: pre-allocated split-KV m/l buffer (workflow bypass)
        out_buf: pre-allocated output buffer [batch, n_heads, head_dim] (graph-safe)

    Returns:
        [batch, n_heads, head_dim] (bf16, 3D)
    """
    _check_available("attn_paged_decode")
    causal_offset = 0 if is_causal else -1
    return _modules["attn_paged_decode"].attn_paged_decode(
        q,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        kv_indptr,
        max_seq_len,
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
    mask: Optional[torch.Tensor] = None,
    max_q_len: int = 0,
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
        req_to_token: [num_reqs, max_context_len] (int64)
        req_pool_indices: [batch] (int64)
        kv_indptr: [batch+1] (int32) — prefix sum of per-request kv_lens
        qo_indptr: [batch+1] (int32) — prefix sum of per-request q_lens
        mask: 4D [batch, 1, q_len, kv_len] (bool, True=keep) or None
        max_q_len: max per-request q_len (Python int, for grid computation)
        is_causal: apply causal mask

    Returns:
        [total_q, n_heads, head_dim] (bf16, 3D)
    """
    _check_available("attn_paged_prefill")
    causal_offset = 0 if is_causal else -1
    return _modules["attn_paged_prefill"].attn_paged_prefill(
        q,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        kv_indptr,
        qo_indptr,
        mask,
        max_q_len,
        causal_offset=causal_offset,
    )
