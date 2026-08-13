"""FP8 linear dispatch: replace aten::linear on the CUDA key, no model changes.

``F.linear`` -> ``aten::linear`` -> dispatcher -> this CUDA impl (fp8 when
enabled) or the original composite implementation via ``redispatch``.
Enabling is per-thread; model code stays untouched.
"""

import threading

import torch
from torch.library import Library

from astrai.extension.fp8_ops import fp8_linear_forward

_state = threading.local()


def fp8_linear_enable(enabled: bool = True) -> None:
    """Toggle fp8 dispatch for aten::linear on this thread."""
    _state.enabled = enabled


def fp8_linear_enabled() -> bool:
    return getattr(_state, "enabled", False)


def _linear_cuda_impl(x: torch.Tensor, w: torch.Tensor, bias=None):
    if fp8_linear_enabled() and x.dtype in (torch.bfloat16, torch.float32):
        return fp8_linear_forward(x, w, bias)
    return torch.ops.aten.linear.default.redispatch(
        torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd),
        x,
        w,
        bias,
    )


def _linear_backward_cuda_impl(input, grad_output, weight, output_mask):
    # VariableType wraps aten::linear; its backward runs aten::linear_backward
    # with schema (self, grad_output, weight, mask). Implement the bf16
    # gradient math directly (no redispatch), supporting [..., K] inputs:
    #   dX = g @ W, dW = g^T @ X, dB = sum(g, dim=0)
    g = grad_output.to(torch.bfloat16)
    g2d = g.reshape(-1, weight.size(0))
    x2d = input.reshape(-1, input.size(-1)).to(torch.bfloat16)
    dX = (
        torch.mm(g2d, weight)
        if output_mask[0]
        else torch.empty(0, device=input.device, dtype=input.dtype)
    )
    dX = dX.reshape_as(input)
    dW = (
        torch.mm(g2d.t(), x2d)
        if output_mask[1]
        else torch.empty(0, device=input.device, dtype=input.dtype)
    )
    dB = (
        g.sum(dim=0)
        if output_mask[2]
        else torch.empty(0, device=input.device, dtype=input.dtype)
    )
    return dX, dW, dB


_lib = Library("aten", "IMPL", "CUDA")
_lib.impl("linear", _linear_cuda_impl)
_lib.impl("linear_backward", _linear_backward_cuda_impl)
