"""Inference-only dispatch for AstrAI linear layers.

The CUDA GEMV path is narrow by construction rather than by a measured
shape table: the kernel streams each weight exactly once, so automatic
selection is keyed on the decode batch size alone (M in [2, 4], where it
sits at the HBM bandwidth floor and beat the cuBLAS small-M path on every
measured family). Every training, prefill-sized, out-of-band, or
unsupported call falls back to PyTorch.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.dispatch import env_mode
from astrai.extension.loader import is_available
from astrai.extension.ops.gemv import bf16_gemv

# M=1 keeps cuBLAS (its GEMV path is already at the bandwidth floor; only
# OPT 1.3B shapes ever passed the full gate). M >= 5 approaches the cuBLAS
# tensor-core crossover (M=8 regressed at wrapper level on every measured
# family, and cuBLAS clearly wins from M ~ 12).
_AUTO_GEMV_M = frozenset({2, 3, 4})


def _inference_bf16_gemv(
    x: Tensor, weight: Tensor, bias: Optional[Tensor] = None
) -> Tensor:
    # Model parameters retain requires_grad=True after eval().  Dispatch is
    # already restricted to no-grad, so detached views preserve storage and
    # layout while satisfying the primitive's explicit autograd guard.
    return bf16_gemv(
        x.detach(),
        weight.detach(),
        bias.detach() if bias is not None else None,
    )


def _gemv_capable(x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> bool:
    if (
        torch.is_grad_enabled()
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or x.ndim not in (1, 2)
        or (x.ndim == 2 and not 1 <= x.shape[0] <= 8)
        or x.shape[-1] != weight.shape[1]
        or x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or torch.cuda.get_device_capability(x.get_device()) < (8, 0)
        or not is_available("bf16_gemv")
    ):
        return False
    return bias is None or (
        bias.device == x.device
        and bias.dtype == torch.bfloat16
        and bias.ndim == 1
        and bias.shape[0] == weight.shape[0]
        and bias.is_contiguous()
    )


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Apply a linear projection with safe inference-only GEMV dispatch.

    ``ASTRAI_GEMV=0`` always uses PyTorch, ``1`` forces GEMV whenever the
    primitive can safely handle any M in ``{1, ..., 8}``, and ``auto`` (the
    default) uses GEMV for decode batches with M in ``{2, 3, 4}``.
    """
    mode = env_mode("ASTRAI_GEMV")
    if mode != "0" and _gemv_capable(x, weight, bias):
        m = 1 if x.ndim == 1 else x.shape[0]
        if mode == "1" or m in _AUTO_GEMV_M:
            return _inference_bf16_gemv(x, weight, bias)
    return F.linear(x, weight, bias)


__all__ = ["linear"]
