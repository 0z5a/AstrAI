"""Rotary embedding dispatch (family "rotary").

Registered rows: the fused CUDA kernel (bf16 CUDA, inference-only) and the
torch complex-multiply fallback (autograd-safe).  Selection runs through
the generic dispatcher, so ``op_backend(rotary=...)`` and
``ASTR_OPS=rotary=torch`` work exactly like for attention.

Layout: x is [batch, seq_len, n_heads, head_dim] (bf16).
freqs_cis is [batch, seq_len, dim/2, 2] (f32) — [cos, sin] pairs.
"""

from typing import Any, Dict, List

import torch
from torch import Tensor

from astrai.extension.dispatch import (
    ImplRecord,
    Spec,
    axis,
    register_family,
    resolve,
    tensor_axes,
)
from astrai.extension.loader import is_available
from astrai.extension.ops.rotary import rotary_emb as _cuda_rotary

_SPEC_CUDA = (
    axis("device_cuda").truthy()
    & axis("dtype").in_(torch.bfloat16)
    & axis("grad_enabled").eq(False)
)


def _torch_apply(x: Tensor, freqs_cis: Tensor) -> Tensor:
    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
    dtype = x.dtype
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_cis_complex = torch.complex(cos, sin).unsqueeze(-2)
    x_rotated = x_complex * freqs_cis_complex
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.to(dtype)


def _rotary_records() -> List[ImplRecord]:
    return [
        ImplRecord(
            family="rotary",
            name="cuda",
            obj=_cuda_rotary,
            spec=_SPEC_CUDA,
            available=lambda: is_available("rotary_emb"),
            priority=0,
        ),
        ImplRecord(
            family="rotary",
            name="torch",
            obj=_torch_apply,
            spec=Spec.always(),
            priority=99,
        ),
    ]


def _axes(x: Tensor, freqs_cis: Tensor) -> Dict[str, Any]:
    return tensor_axes(x)


def _fallback_record() -> ImplRecord:
    return _rotary_records()[-1]


register_family("rotary", _axes, _rotary_records, _fallback_record)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply rotary embedding to x.

    Args:
        x: [batch, seq_len, n_heads, head_dim] (bf16)
        freqs_cis: [batch, seq_len, dim/2, 2] (f32) — [cos, sin] pairs

    Returns:
        [batch, seq_len, n_heads, head_dim] (bf16)
    """
    return resolve("rotary", x, freqs_cis).record.obj(x, freqs_cis)


__all__ = ["apply_rotary_emb"]
