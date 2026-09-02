"""Inference-only fused SwiGLU selection for dense MLP layers."""

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.backend.linear import linear
from astrai.extension.dispatch import env_mode
from astrai.extension.loader import is_available
from astrai.extension.ops.swiglu import bf16_swiglu


def _unfused_swiglu(x: Tensor, up_weight: Tensor, gate_weight: Tensor) -> Tensor:
    # Keep the existing linear backend in the fallback chain. This preserves
    # any independently qualified GEMV batches instead of making the fusion
    # decision suppress linear-level optimizations.
    return linear(x, up_weight) * F.silu(linear(x, gate_weight))


def _fused_swiglu(x: Tensor, up_weight: Tensor, gate_weight: Tensor) -> Tensor:
    return bf16_swiglu(x.detach(), up_weight.detach(), gate_weight.detach())


def _swiglu_capable(x: Tensor, up_weight: Tensor, gate_weight: Tensor) -> bool:
    return not (
        torch.is_grad_enabled()
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or up_weight.dtype != torch.bfloat16
        or gate_weight.dtype != torch.bfloat16
        or x.ndim not in (1, 2)
        or up_weight.ndim != 2
        or gate_weight.ndim != 2
        or (x.ndim == 2 and not 1 <= x.shape[0] <= 8)
        or up_weight.shape != gate_weight.shape
        or x.shape[-1] != up_weight.shape[1]
        or x.shape[-1] % 8 != 0
        or x.device != up_weight.device
        or x.device != gate_weight.device
        or not x.is_contiguous()
        or not up_weight.is_contiguous()
        or not gate_weight.is_contiguous()
        or not is_available("bf16_swiglu")
    )


def swiglu(x: Tensor, up_weight: Tensor, gate_weight: Tensor) -> Tensor:
    """Apply the dense-MLP SwiGLU projection with a safe torch fallback.

    ``ASTRAI_SWIGLU=0`` and ``auto`` keep the unfused linear-backend chain;
    ``1`` forces the fused primitive for supported inputs. Auto will adopt
    an M-banded rule mirroring the linear backend once end-to-end evidence
    qualifies one.
    """
    if env_mode("ASTRAI_SWIGLU") != "1" or not _swiglu_capable(
        x, up_weight, gate_weight
    ):
        return _unfused_swiglu(x, up_weight, gate_weight)
    return _fused_swiglu(x, up_weight, gate_weight)


__all__ = ["swiglu"]
