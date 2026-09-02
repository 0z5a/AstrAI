"""Stateless wrapper for the directly callable fused BF16 SwiGLU primitive."""

import torch

from astrai.extension.loader import get_module


def bf16_swiglu(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute ``linear(x, up) * silu(linear(x, gate))`` for M in [1, 8].

    Inputs must be contiguous BF16 CUDA tensors. Both weights use row-major
    ``[N, K]`` storage with identical shapes, and K must be divisible by 8.
    The primitive is inference-only and performs no fallback.
    """
    return get_module("bf16_swiglu").bf16_swiglu(x, up_weight, gate_weight)


__all__ = ["bf16_swiglu"]
