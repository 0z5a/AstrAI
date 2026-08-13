"""FP8 linear dispatch: replace aten::linear on the CUDA key, no model changes.

``F.linear`` -> ``aten::linear`` -> dispatcher -> this CUDA impl (fp8 when
enabled) or the original composite implementation via ``redispatch``.
Enabling is per-thread; model code stays untouched.
"""

import threading

import torch
from torch.library import Library

from astrai.extension.fp8_ops import fp8_linear_backward, fp8_linear_forward

_state = threading.local()


def fp8_linear_enable(enabled: bool = True) -> None:
    """Toggle fp8 dispatch for aten::linear on this thread."""
    _state.enabled = enabled


def fp8_linear_enabled() -> bool:
    return getattr(_state, "enabled", False)


def _linear_cuda_impl(x: torch.Tensor, w: torch.Tensor, bias=None):
    if fp8_linear_enabled() and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
        return fp8_linear_forward(x, w, bias)
    return torch.ops.aten.linear.default.redispatch(
        torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd),
        x,
        w,
        bias,
    )


def _linear_backward_cuda_impl(input_tensor, grad_output, weight, output_mask):
    # VariableType wraps aten::linear; its backward runs aten::linear_backward
    # with schema (self, grad_output, weight, mask). When fp8 is enabled the
    # fused CUDA backward runs in one call (scale-corrected); otherwise the
    # plain bf16/fp32 math, dtype aligned to the leaf weight:
    #   grad_input = g @ W, grad_weight = g^T @ X, grad_bias = sum(g, dim=0)
    if fp8_linear_enabled() and weight.dtype == torch.bfloat16:
        return fp8_linear_backward(grad_output, input_tensor, weight, list(output_mask))
    compute_dtype = weight.dtype
    grad = grad_output.to(compute_dtype)
    grad_2d = grad.reshape(-1, weight.size(0))
    input_2d = input_tensor.reshape(-1, input_tensor.size(-1)).to(compute_dtype)
    grad_input = (
        torch.mm(grad_2d, weight)
        if output_mask[0]
        else torch.empty(0, device=input_tensor.device, dtype=input_tensor.dtype)
    )
    grad_weight = (
        torch.mm(grad_2d.t(), input_2d)
        if output_mask[1]
        else torch.empty(0, device=input_tensor.device, dtype=input_tensor.dtype)
    )
    grad_bias = (
        grad.sum(dim=0)
        if output_mask[2]
        else torch.empty(0, device=input_tensor.device, dtype=input_tensor.dtype)
    )
    return grad_input.reshape_as(input_tensor), grad_weight, grad_bias


_lib = Library("aten", "IMPL", "CUDA")
_lib.impl("linear", _linear_cuda_impl)
_lib.impl("linear_backward", _linear_backward_cuda_impl)
