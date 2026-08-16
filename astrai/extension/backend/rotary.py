"""Rotary embedding with auto-dispatch to CUDA kernel.

Single entry point ``apply_rotary_emb(x, freqs_cis)`` — uses the fused
CUDA kernel when available, falls back to torch complex multiply otherwise.

Layout: x is [batch, seq_len, n_heads, head_dim] (bf16).
freqs_cis is [batch, seq_len, dim/2, 2] (f32) — [cos, sin] pairs.
"""

import torch
from torch import Tensor

from astrai.extension.loader import is_available
from astrai.extension.ops.rotary import rotary_emb as _cuda_rotary

_cache = {"available": None}


def _cuda_available() -> bool:
    if _cache["available"] is None:
        _cache["available"] = is_available("rotary_emb")
    return _cache["available"]


def _torch_apply(x: Tensor, freqs_cis: Tensor) -> Tensor:
    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
    dtype = x.dtype
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_cis_complex = torch.complex(cos, sin).unsqueeze(-2)
    x_rotated = x_complex * freqs_cis_complex
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.to(dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply rotary embedding to x.

    Args:
        x: [batch, seq_len, n_heads, head_dim] (bf16)
        freqs_cis: [batch, seq_len, dim/2, 2] (f32) — [cos, sin] pairs

    Returns:
        [batch, seq_len, n_heads, head_dim] (bf16)
    """
    if (
        _cuda_available()
        and not torch.is_grad_enabled()
        and x.is_cuda
        and x.dtype == torch.bfloat16
    ):
        return _cuda_rotary(x, freqs_cis)
    return _torch_apply(x, freqs_cis)
