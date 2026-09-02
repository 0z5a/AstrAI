"""Inference-only dispatch for AstrAI linear layers.

The CUDA GEMV path is deliberately narrow: automatic selection is enabled
only for single-row BF16 shapes measured to beat ``F.linear`` on a supported
architecture. Every training, prefill, unsupported-layout, and unmeasured
call falls back to PyTorch.
"""

import logging
import os
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.dispatch import (
    ImplRecord,
    Spec,
    axis,
    get_override,
    register_family,
    resolve,
    tensor_axes,
)
from astrai.extension.loader import is_available
from astrai.extension.ops.gemv import bf16_gemv

logger = logging.getLogger(__name__)

# Shape keys are (N, K) for Y[M, N] = X[M, K] @ W[N, K].T. A band is
# automatic only after both the per-shape >=5% and end-to-end decode >=3%
# gates pass. L20 micro-winners did not produce a stable whole-graph win, so
# SM89 intentionally has no automatic entries yet. Mode 1 remains available
# for explicit A/B runs without weakening the default.
_AUTO_GEMV_SHAPES: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}

_VALID_MODES = {"0", "1", "auto"}
_WARNED_MODES: set[str] = set()


def _gemv_mode() -> str:
    mode = os.environ.get("ASTRAI_GEMV", "auto").strip().lower()
    if mode in _VALID_MODES:
        return mode
    if mode not in _WARNED_MODES:
        _WARNED_MODES.add(mode)
        logger.warning(
            "ASTRAI_GEMV=%r is invalid; expected 0, 1, or auto; using auto",
            mode,
        )
    return "auto"


def _axes(
    x: Tensor, weight: Tensor, bias: Optional[Tensor] = None
) -> dict[str, object]:
    x_shape = tuple(x.shape)
    weight_shape = tuple(weight.shape)
    single_row = x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1)
    shape_matches = (
        weight.ndim == 2
        and x.ndim in (1, 2)
        and bool(x_shape)
        and x_shape[-1] == weight_shape[-1]
    )
    same_device = x.device == weight.device and (
        bias is None or bias.device == x.device
    )
    bias_supported = bias is None or (
        bias.ndim == 1
        and weight.ndim == 2
        and bias.shape[0] == weight.shape[0]
        and bias.dtype == torch.bfloat16
        and bias.is_contiguous()
    )
    capability = torch.cuda.get_device_capability(x.device) if x.is_cuda else None
    n = weight_shape[0] if weight.ndim == 2 else None
    k = weight_shape[1] if weight.ndim == 2 else None
    return tensor_axes(
        x,
        mode=_gemv_mode(),
        capability=capability,
        n=n,
        k=k,
        single_row=single_row,
        shape_matches=shape_matches,
        same_device=same_device,
        weight_dtype=weight.dtype,
        x_contiguous=x.is_contiguous(),
        weight_contiguous=weight.is_contiguous(),
        bias_supported=bias_supported,
        k_even=k is not None and k % 2 == 0,
    )


_SPEC_CAPABLE = (
    axis("device_cuda").truthy()
    & axis("dtype").in_(torch.bfloat16)
    & axis("weight_dtype").in_(torch.bfloat16)
    & axis("grad_enabled").eq(False)
    & axis("single_row").truthy()
    & axis("shape_matches").truthy()
    & axis("same_device").truthy()
    & axis("x_contiguous").truthy()
    & axis("weight_contiguous").truthy()
    & axis("bias_supported").truthy()
    & axis("k_even").truthy()
)

_SPEC_AUTO = _SPEC_CAPABLE & Spec.of(
    lambda ax: (
        (ax.get("n"), ax.get("k")) in _AUTO_GEMV_SHAPES.get(ax.get("capability"), ())
    ),
    "shape is a measured winner for this architecture",
)


def _torch_linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    return F.linear(x, weight, bias)


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


@lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _gemv_capable(x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> bool:
    if (
        torch.is_grad_enabled()
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or x.ndim not in (1, 2)
        or (x.ndim == 2 and x.shape[0] != 1)
        or x.shape[-1] != weight.shape[1]
        or weight.shape[1] % 2 != 0
        or x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
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


def _auto_gemv_shape(x: Tensor, weight: Tensor) -> bool:
    capability = _device_capability(x.get_device())
    return (weight.shape[0], weight.shape[1]) in _AUTO_GEMV_SHAPES.get(capability, ())


def _linear_records() -> list[ImplRecord]:
    mode = _gemv_mode()
    gemv_priority = 0 if mode == "1" else 100
    auto_priority = 0 if mode == "auto" else 90
    torch_priority = 0 if mode == "0" else 50
    return [
        ImplRecord(
            family="linear",
            name="gemv",
            obj=_inference_bf16_gemv,
            spec=_SPEC_CAPABLE,
            available=lambda: is_available("bf16_gemv"),
            priority=gemv_priority,
        ),
        ImplRecord(
            family="linear",
            name="auto_gemv",
            obj=_inference_bf16_gemv,
            spec=_SPEC_AUTO,
            available=lambda: is_available("bf16_gemv"),
            priority=auto_priority,
        ),
        ImplRecord(
            family="linear",
            name="torch",
            obj=_torch_linear,
            spec=Spec.always(),
            priority=torch_priority,
        ),
    ]


def _fallback_record() -> ImplRecord:
    return ImplRecord(
        family="linear",
        name="torch",
        obj=_torch_linear,
        spec=Spec.always(),
        priority=999,
    )


register_family("linear", _axes, _linear_records, _fallback_record)


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Apply a linear projection with safe inference-only GEMV dispatch.

    ``ASTRAI_GEMV=0`` always uses PyTorch, ``1`` forces GEMV whenever the
    primitive can safely handle the call, and ``auto`` (the default) uses
    only architecture/shape bands backed by benchmark evidence.
    """
    # Preserve the shared dispatcher for explicit/context selection and
    # ASTR_OPS diagnostics, while keeping the default per-layer hot path free
    # of axes dictionaries, record sorting, and repeated capability queries.
    if get_override("linear") is not None or "linear" in os.environ.get("ASTR_OPS", ""):
        return resolve("linear", x, weight, bias).record.obj(x, weight, bias)

    mode = _gemv_mode()
    if mode == "0" or (mode == "auto" and not _AUTO_GEMV_SHAPES):
        return _torch_linear(x, weight, bias)
    if mode != "0" and _gemv_capable(x, weight, bias):
        if mode == "1" or _auto_gemv_shape(x, weight):
            return _inference_bf16_gemv(x, weight, bias)
    return _torch_linear(x, weight, bias)


__all__ = ["linear"]
