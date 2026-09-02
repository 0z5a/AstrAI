"""Stateless wrapper for the directly callable BF16 GEMV primitive."""

from typing import Optional

import torch

from astrai.extension.loader import get_module


def bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``F.linear(x, weight, bias)`` for up to eight BF16 rows.

    ``x`` must have shape ``[K]`` or ``[M, K]`` with M in ``[1, 8]``,
    and ``weight`` must be a contiguous row-major ``[N, K]`` tensor. The CUDA
    kernel reuses each weight row across M, accumulates in FP32, and returns
    BF16. This primitive is inference-only and intentionally performs no
    fallback or model-level dispatch.
    """
    return get_module("bf16_gemv").bf16_gemv(x, weight, bias)
