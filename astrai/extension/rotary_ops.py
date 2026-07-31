"""Rotary embedding CUDA kernel wrapper.

Calls the compiled CUDA kernel directly. If the kernel is not available,
raises ``RuntimeError``. Fallback to torch complex multiply is the
responsibility of ``astrai.extension.rotary_backend.apply_rotary_emb``.

Layout: x is [batch, seq_len, n_heads, head_dim] (bf16, contiguous).
freqs_cis is [batch, seq_len, head_dim/2, 2] (f32, contiguous) — [cos, sin] pairs.
"""

import torch

from astrai.extension.loader import _available, _modules


def _check_available():
    if not _available.get("rotary_emb"):
        raise RuntimeError(
            "CUDA kernel 'rotary_emb' is not available. "
            "Build with CSRC_KERNELS=true or use the torch fallback."
        )


def rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Fused rotary embedding kernel.

    Args:
        x: [batch, seq_len, n_heads, head_dim] (bf16, contiguous)
        freqs_cis: [batch, seq_len, head_dim/2, 2] (f32, contiguous) — [cos, sin] pairs

    Returns:
        [batch, seq_len, n_heads, head_dim] (bf16)
    """
    _check_available()
    if not x.is_contiguous():
        x = x.contiguous()
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()
    return _modules["rotary_emb"].rotary_emb(x, freqs_cis)
