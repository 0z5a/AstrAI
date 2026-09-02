"""Inference-only fused SwiGLU selection for dense MLP layers."""

import logging
import os
from functools import cache

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.backend.linear import linear
from astrai.extension.loader import is_available
from astrai.extension.ops.swiglu import bf16_swiglu

logger = logging.getLogger(__name__)

# Shape keys are (N, K) for the paired up/gate projections. Automatic entries
# are populated only after the primitive, MLP chain, and greedy checkpoint
# gates pass on that architecture.
_AUTO_SWIGLU_SHAPES: dict[tuple[int, int], dict[int, frozenset[tuple[int, int]]]] = {}
_AUTO_SWIGLU_M = frozenset(
    m for architecture in _AUTO_SWIGLU_SHAPES.values() for m in architecture
)
_VALID_MODES = {"0", "1", "auto"}
_WARNED_MODES: set[str] = set()


def _swiglu_mode() -> str:
    mode = os.environ.get("ASTRAI_SWIGLU", "auto").strip().lower()
    if mode in _VALID_MODES:
        return mode
    if mode not in _WARNED_MODES:
        _WARNED_MODES.add(mode)
        logger.warning(
            "ASTRAI_SWIGLU=%r is invalid; expected 0, 1, or auto; using auto",
            mode,
        )
    return "auto"


def _unfused_swiglu(x: Tensor, up_weight: Tensor, gate_weight: Tensor) -> Tensor:
    # Keep the existing linear backend in the fallback chain. This preserves
    # any independently qualified GEMV shapes instead of making the fusion
    # decision suppress linear-level optimizations.
    return linear(x, up_weight) * F.silu(linear(x, gate_weight))


def _fused_swiglu(x: Tensor, up_weight: Tensor, gate_weight: Tensor) -> Tensor:
    return bf16_swiglu(x.detach(), up_weight.detach(), gate_weight.detach())


@cache
def _device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


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


def _auto_swiglu_shape(x: Tensor, up_weight: Tensor) -> bool:
    capability = _device_capability(x.get_device())
    m = 1 if x.ndim == 1 else x.shape[0]
    return (up_weight.shape[0], up_weight.shape[1]) in _AUTO_SWIGLU_SHAPES.get(
        capability, {}
    ).get(m, ())


def swiglu(x: Tensor, up_weight: Tensor, gate_weight: Tensor) -> Tensor:
    """Apply the dense-MLP SwiGLU projection with a safe torch fallback.

    ``ASTRAI_SWIGLU=0`` keeps the unfused linear-backend chain, ``1`` forces
    the fused primitive for supported inputs, and ``auto`` uses only
    architecture/shape bands backed by benchmark and checkpoint evidence.
    """
    mode = _swiglu_mode()
    if mode == "0" or (mode == "auto" and not _AUTO_SWIGLU_SHAPES):
        return _unfused_swiglu(x, up_weight, gate_weight)
    if mode == "auto":
        m = 1 if x.ndim == 1 else (x.shape[0] if x.ndim == 2 else None)
        if m not in _AUTO_SWIGLU_M:
            return _unfused_swiglu(x, up_weight, gate_weight)
    if _swiglu_capable(x, up_weight, gate_weight) and (
        mode == "1" or _auto_swiglu_shape(x, up_weight)
    ):
        return _fused_swiglu(x, up_weight, gate_weight)
    return _unfused_swiglu(x, up_weight, gate_weight)


__all__ = ["swiglu"]
