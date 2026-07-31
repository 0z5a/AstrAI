"""Rotary embedding with auto-dispatch to CUDA kernel.

Single entry point ``apply_rotary_emb(x, cos, sin)`` — uses the fused
CUDA kernel when available, falls back to torch complex multiply otherwise.

Layout: x is [batch, seq_len, n_heads, head_dim] (bf16).
cos/sin are [batch, seq_len, head_dim/2] (f32).
"""

import torch
from torch import Tensor

from astrai.extension.loader import is_available
from astrai.extension.rotary_ops import rotary_emb as _cuda_rotary

_cache = {"available": None}


def _cuda_available() -> bool:
    if _cache["available"] is None:
        _cache["available"] = is_available("rotary_emb")
    return _cache["available"]


def _torch_apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    dtype = x.dtype
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_cis = torch.complex(cos, sin).unsqueeze(2)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.to(dtype)


def apply_rotary_emb(x: Tensor, rotary_emb: tuple[Tensor, Tensor]) -> Tensor:
    """Apply rotary embedding to x.

    Args:
        x: [batch, seq_len, n_heads, head_dim] (bf16)
        rotary_emb: (cos, sin) tuple, each [batch, seq_len, head_dim/2] (f32)

    Returns:
        [batch, seq_len, n_heads, head_dim] (bf16)
    """
    cos, sin = rotary_emb
    if _cuda_available() and x.is_cuda and x.dtype == torch.bfloat16:
        return _cuda_rotary(x, cos, sin)
    return _torch_apply(x, cos, sin)
