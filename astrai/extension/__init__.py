"""CUDA attention kernel wrappers with torch fallback.

Public API:
    - ``attn_decode`` — single-query decode attention
    - ``attn_prefill`` — multi-query prefill attention
    - ``attn_paged_decode`` — paged decode attention (direct page-table access)
    - ``AttentionBackend`` — ABC for attention computation strategies
    - ``TorchNativeBackend`` — default SDPA backend with KV cache I/O

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). Scale is always ``1/sqrt(head_dim)``.

Each wrapper dispatches to its compiled CUDA kernel (``astrai.extension.attn_*``)
when available, otherwise falls back to ``torch.nn.functional.scaled_dot_product_attention``.
"""

from astrai.extension.attention_backend import (
    ATTN_BACKEND,
    AttentionBackend,
    TorchNativeBackend,
    attn_backend,
    get_backend,
)
from astrai.extension.attention_ops import (
    attn_decode,
    attn_paged_decode,
    attn_prefill,
)
from astrai.extension.loader import KERNEL_NAMES, is_available

__all__ = [
    "ATTN_BACKEND",
    "AttentionBackend",
    "TorchNativeBackend",
    "attn_backend",
    "get_backend",
    "attn_decode",
    "attn_paged_decode",
    "attn_prefill",
    "is_available",
    "KERNEL_NAMES",
]
