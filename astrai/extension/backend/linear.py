"""Inference-only dispatch for AstrAI linear layers.

The CUDA GEMV path is narrow by construction rather than by a measured
shape table: the kernel streams each weight exactly once, so automatic
selection is keyed on the decode batch size alone (M in [2, 4], where it
sits at the HBM bandwidth floor and beat the cuBLAS small-M path on every
measured family). Every training, prefill-sized, out-of-band, or
unsupported call falls back to PyTorch.

The family stays registered with the shared operator dispatcher, so
``op_backend(linear=...)``, ``ASTR_OPS=linear=...``, and ``resolve`` /
``explain`` keep working like for attention and rotary. The per-layer
hot path only consults the dispatcher when one of those selections is
active, keeping it free of axes dictionaries and record sorting.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.dispatch import (
    ImplRecord,
    Spec,
    axis,
    env_mode,
    env_selection,
    get_override,
    register_family,
    resolve,
    tensor_axes,
)
from astrai.extension.loader import is_available
from astrai.extension.ops.gemv import bf16_gemv

# M=1 keeps cuBLAS (its GEMV path is already at the bandwidth floor; only
# OPT 1.3B shapes ever passed the full gate). M >= 5 approaches the cuBLAS
# tensor-core crossover (M=8 regressed at wrapper level on every measured
# family, and cuBLAS clearly wins from M ~ 12).
_AUTO_GEMV_M = frozenset({2, 3, 4})


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


def _axes(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Dict[str, Any]:
    weight_shape = tuple(weight.shape)
    m = 1 if x.ndim == 1 else (x.shape[0] if x.ndim == 2 else None)
    supported_m = m is not None and 1 <= m <= 8
    shape_matches = (
        weight.ndim == 2
        and x.ndim in (1, 2)
        and bool(x.shape)
        and x.shape[-1] == weight_shape[-1]
    )
    same_device = x.device == weight.device and (
        bias is None or bias.device == x.device
    )
    bias_supported = bias is None or (
        bias.ndim == 1
        and weight.ndim == 2
        and bias.shape[0] == weight_shape[0]
        and bias.dtype == torch.bfloat16
        and bias.is_contiguous()
    )
    capability = torch.cuda.get_device_capability(x.device) if x.is_cuda else None
    return tensor_axes(
        x,
        mode=env_mode("ASTRAI_GEMV"),
        m=m,
        supported_m=supported_m,
        auto_m=m in _AUTO_GEMV_M,
        shape_matches=shape_matches,
        same_device=same_device,
        weight_dtype=weight.dtype,
        x_contiguous=x.is_contiguous(),
        weight_contiguous=weight.is_contiguous(),
        bias_supported=bias_supported,
        capability=capability,
    )


_SPEC_CAPABLE = (
    axis("device_cuda").truthy()
    & axis("dtype").in_(torch.bfloat16)
    & axis("weight_dtype").in_(torch.bfloat16)
    & axis("grad_enabled").eq(False)
    & axis("supported_m").truthy()
    & axis("shape_matches").truthy()
    & axis("same_device").truthy()
    & axis("x_contiguous").truthy()
    & axis("weight_contiguous").truthy()
    & axis("bias_supported").truthy()
    & Spec.of(
        lambda ax: ax.get("capability") is not None and ax.get("capability") >= (8, 0),
        "capability>=sm_80",
    )
)

_SPEC_AUTO = _SPEC_CAPABLE & axis("auto_m").truthy()


def _linear_records() -> List[ImplRecord]:
    mode = env_mode("ASTRAI_GEMV")
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
    primitive can safely handle any M in ``{1, ..., 8}``, and ``auto`` (the
    default) uses GEMV for decode batches with M in ``{2, 3, 4}``.
    """
    # Route through the shared dispatcher whenever a selection is active so
    # explicit/context/env overrides stay honored; otherwise keep the hot
    # path free of axes dictionaries and record sorting.
    if get_override("linear") is not None or env_selection("linear") is not None:
        return resolve("linear", x, weight, bias).record.obj(x, weight, bias)

    mode = env_mode("ASTRAI_GEMV")
    if mode != "0" and _gemv_capable(x, weight, bias):
        m = 1 if x.ndim == 1 else x.shape[0]
        if mode == "1" or m in _AUTO_GEMV_M:
            return _inference_bf16_gemv(x, weight, bias)
    return _torch_linear(x, weight, bias)


__all__ = ["linear"]
