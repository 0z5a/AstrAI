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

import torch

from astrai.extension.loader import _available, _modules


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
    mask: torch.Tensor | None = None,
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
        q, k, v, mask=mask, causal_offset=causal_offset, layout=1
    )


def attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
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
        q, k, v, mask=mask, causal_offset=causal_offset, layout=1
    )


def attn_paged_decode(
    q: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_size: int,
    kv_len: int,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Paged GQA decode attention (q_len == 1, direct page-table access).

    Args:
        q: [batch, 1, n_heads, head_dim] (blhd, bf16)
        page_table: [batch, max_pages] (int64)
        k_cache: [n_pages, page_size, n_kv_heads, head_dim] (bf16)
        v_cache: same as k_cache
        page_size: tokens per page
        kv_len: actual sequence length per request
        mask: 2D [batch, kv_len] or 3D [batch, 1, kv_len] (bool, True=keep)
        is_causal: apply causal mask

    Returns:
        [batch, 1, n_heads, head_dim] (blhd, bf16)
    """
    _check_available("attn_paged_decode")
    causal_offset = (kv_len - 1) if is_causal else -1
    return _modules["attn_paged_decode"].attn_paged_decode(
        q,
        page_table,
        k_cache,
        v_cache,
        page_size,
        kv_len,
        mask=mask,
        causal_offset=causal_offset,
        layout=1,
    )
