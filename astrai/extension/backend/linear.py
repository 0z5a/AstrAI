"""Inference-only dispatch for AstrAI linear layers.

The CUDA GEMM path is sized by the decode batch M. M in [1, 8] uses the
register-resident GEMV kernel (any K); M in (8, 64] uses the tiled
kernel (K % 8 == 0, 16-byte-aligned tensors). Automatic mode
selects GEMM for M in [1, 64] where the primitive is capable; every
training, prefill-sized, or unsupported call falls back to PyTorch.

The family stays registered with the shared operator dispatcher, so
``op_backend(linear=...)``, ``ASTR_OPS=linear=...``, and ``resolve`` /
``explain`` keep working. The per-layer hot path only consults the
dispatcher when one of those selections is active.
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
from astrai.extension.ops.gemm import bf16_gemm


def _torch_linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    return F.linear(x, weight, bias)


def _inference_bf16_gemm(
    x: Tensor, weight: Tensor, bias: Optional[Tensor] = None
) -> Tensor:
    return bf16_gemm(
        x.detach(),
        weight.detach(),
        bias.detach() if bias is not None else None,
    )


def _gemm_capable(x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> bool:
    """Check whether bf16_gemm can safely handle the call."""
    if (
        torch.is_grad_enabled()
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or x.ndim not in (1, 2)
        or x.shape[-1] != weight.shape[1]
        or x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or torch.cuda.get_device_capability(x.get_device()) < (8, 0)
        or not is_available("bf16_gemm")
    ):
        return False
    m = 1 if x.ndim == 1 else x.shape[0]
    # M <= 32 is where the kernel wins: L2-rotation measurements on L20
    # show every production shape at M=24-32 winning or tying, while
    # M=48-64 loses the long-K down_proj by 8-10% (cuBLAS switches to a
    # wider tile there). The kernel itself still accepts M <= 64 when
    # called directly through astrai.extension.ops.gemm.
    if not (1 <= m <= 32):
        return False
    # Vocabulary-sized lm_head weights (N in the tens of thousands+) stream
    # better through cuBLAS: our skinny path ties it at M<=8 and the tiled
    # path loses ~4% at M=9-16 (L2-rotation measurements on L20). Gate the
    # whole shape family out instead of splitting hairs per M band.
    if weight.shape[0] > 32768:
        return False
    # M > 8 tiled path requires K % 8 == 0 and 16-byte alignment.
    k = x.shape[-1]
    if m > 8 and (
        k % 8 != 0 or (x.data_ptr() & 15) != 0 or (weight.data_ptr() & 15) != 0
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
    supported_m = m is not None and 1 <= m <= 64
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
        mode=env_mode("ASTRAI_GEMM"),
        m=m,
        supported_m=supported_m,
        auto_m=supported_m,
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
    mode = env_mode("ASTRAI_GEMM")
    gemm_priority = 0 if mode == "1" else 100
    auto_priority = 0 if mode == "auto" else 90
    torch_priority = 0 if mode == "0" else 50
    return [
        ImplRecord(
            family="linear",
            name="gemm",
            obj=_inference_bf16_gemm,
            spec=_SPEC_CAPABLE,
            available=lambda: is_available("bf16_gemm"),
            priority=gemm_priority,
        ),
        ImplRecord(
            family="linear",
            name="auto_gemm",
            obj=_inference_bf16_gemm,
            spec=_SPEC_AUTO,
            available=lambda: is_available("bf16_gemm"),
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
    """Apply a linear projection with safe inference-only GEMM dispatch.

    ``ASTRAI_GEMM=0`` always uses PyTorch, ``1`` forces GEMM whenever the
    primitive can safely handle the call (M in [1, 64], K % 8 == 0 and
    16-byte-aligned for M > 8), and ``auto`` (the default) selects GEMM
    for all capable decode batches.
    """
    if get_override("linear") is not None or env_selection("linear") is not None:
        return resolve("linear", x, weight, bias).record.obj(x, weight, bias)

    mode = env_mode("ASTRAI_GEMM")
    if mode != "0" and _gemm_capable(x, weight, bias):
        return _inference_bf16_gemm(x, weight, bias)
    return _torch_linear(x, weight, bias)


__all__ = ["linear"]
