"""Rotary embedding CUDA kernel wrapper.

Calls the compiled CUDA kernel directly. If the kernel is not available,
raises ``RuntimeError``. Fallback to torch complex multiply is the
responsibility of ``astrai.extension.backend.rotary.apply_rotary_emb``.

Layout: x is packed [tokens, n_heads, head_dim] or dense
[batch, seq_len, n_heads, head_dim]. ``freqs_cis`` has matching token axes.
"""

import torch

from astrai.extension.loader import get_module


def rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Fused rotary embedding kernel.

    Args:
        x: packed 3D or dense 4D bf16 tensor.
        freqs_cis: matching token axes followed by [head_dim/2, 2].

    Returns:
        Tensor with the same shape as ``x``.
    """
    mod = get_module("rotary_emb")
    if not x.is_contiguous():
        x = x.contiguous()
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()
    return mod.rotary_emb(x, freqs_cis)


__all__ = ["rotary_emb"]
