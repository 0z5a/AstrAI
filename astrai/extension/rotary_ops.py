"""Rotary embedding CUDA kernel wrapper.

Calls the compiled CUDA kernel directly. If the kernel is not available,
raises ``RuntimeError``. Fallback to torch complex multiply is the
responsibility of ``astrai.model.components.rope.apply_rotary_emb``.

Layout convention: x is ``[batch, seq_len, n_heads, head_dim]`` (blhd, bf16).
cos/sin are ``[batch, seq_len, head_dim/2]`` (f32).
"""

import torch

from astrai.extension.loader import _available, _modules


def _check_available():
    if not _available.get("rotary_emb"):
        raise RuntimeError(
            "CUDA kernel 'rotary_emb' is not available. "
            "Build with CSRC_KERNELS=true or use the torch fallback."
        )


def rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Fused rotary embedding kernel.

    Applies rotation: for each pair (x_even, x_odd):
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

    Args:
        x: [batch, seq_len, n_heads, head_dim] (bf16, contiguous)
        cos: [batch, seq_len, head_dim/2] (f32)
        sin: [batch, seq_len, head_dim/2] (f32)

    Returns:
        [batch, seq_len, n_heads, head_dim] (bf16)
    """
    _check_available()
    if not x.is_contiguous():
        x = x.contiguous()
    return _modules["rotary_emb"].rotary_emb(x, cos, sin)
