"""FP8 CUDA kernel interface adapter (the only module touching the pybind.

Isolates the ``fp8_mm`` CUDA extension behind stable Python functions:
- availability / dtype checks and clear errors
- torch.library ``custom::fp8_mm`` registration (meta + CPU fallback)
- quantize-in-GEMM primitives used by ``fp8.py`` training state

Policy (scales, amax history, delayed scaling, autocast) lives in ``fp8.py``;
this module is stateless.
"""

import torch
from torch.library import custom_op

from astrai.extension.loader import get_module, is_available


def _mod():
    if not is_available("fp8_mm"):
        raise RuntimeError(
            "CUDA kernel 'fp8_mm' is not available. Build with CSRC_KERNELS=true."
        )
    return get_module("fp8_mm")


@custom_op("custom::fp8_mm", mutates_args=())
def fp8_mm(
    a: torch.Tensor, b: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor
) -> torch.Tensor:
    """FP8 e4m3 GEMM: a[M,K] x b[N,K] -> bf16[M,N] (pre-scaled inputs)."""


@fp8_mm.register_fake
def _fp8_mm_fake(a, b, sx, sw):
    return torch.empty((a.size(0), b.size(1)), device=a.device, dtype=torch.bfloat16)


@fp8_mm.register_kernel("cuda")
def _fp8_mm_cuda(a, b, sx, sw):
    return _mod().fp8_mm(a, b)


@fp8_mm.register_kernel("cpu")
def _fp8_mm_cpu(a, b, sx, sw):
    return torch.mm(a.float(), b.float().t()).to(torch.bfloat16)


def linear_forward_scaled(x, w, bias, sx, sw, sx_inv, sw_inv, amax_x, amax_w):
    """Quantize x/w with per-tensor scales + cuBLASLt GEMM + bias -> bf16.

    x/w: [..., K] / [N, K] bf16; sx/sw: f32 scale tensors (device scalars);
    sx_inv/sw_inv: 1/scale; amax_x/amax_w: f32 buffers receiving max-abs.
    """
    if not (x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16):
        raise TypeError(f"fp8 forward requires bf16 inputs, got {x.dtype}/{w.dtype}")
    return _mod().fp8_linear_forward_scaled(
        x, w, bias, sx, sw, sx_inv, sw_inv, amax_x, amax_w
    )


def linear_backward_scaled(g, x, w, masks, sg, sw, sx, sg_inv, sw_inv, sx_inv, amax_g):
    """dX = g @ W, dW = g^T @ X, dB = sum(g) with per-tensor scales."""
    if not (
        g.dtype == torch.bfloat16
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
    ):
        raise TypeError(
            f"fp8 backward requires bf16 inputs, got {g.dtype}/{x.dtype}/{w.dtype}"
        )
    return _mod().fp8_linear_backward_scaled(
        g, x, w, masks, sg, sw, sx, sg_inv, sw_inv, sx_inv, amax_g
    )
