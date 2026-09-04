"""Stateless wrapper for the directly callable BF16 GEMM primitive."""

from typing import Optional

import torch

from astrai.extension.loader import get_module


def bf16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``F.linear(x, weight, bias)`` for up to 64 BF16 rows.

    ``x`` must have shape ``[K]`` or ``[M, K]`` with M in ``[1, 64]``, and
    ``weight`` must be a contiguous row-major ``[N, K]`` tensor. M in
    ``[1, 8]`` uses the register-resident skinny GEMM kernel (any K);
    larger M uses the tiled kernel (K must be a multiple of 8 with
    16-byte-aligned tensors). This primitive is inference-only and
    intentionally performs no fallback or model-level dispatch.
    """
    return get_module("bf16_gemm").bf16_gemm(x, weight, bias)


__all__ = ["bf16_gemm"]
