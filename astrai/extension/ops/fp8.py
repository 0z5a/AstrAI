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
    """BF16 inputs, fused FP8 GEMM with FP32 accumulation and BF16 output."""


@fp8_mm.register_fake
def _fp8_mm_fake(a, b, sx, sw):
    return torch.empty((a.size(0), b.size(0)), device=a.device, dtype=torch.bfloat16)


@fp8_mm.register_kernel("cuda")
def _fp8_mm_cuda(a, b, sx, sw):
    if not (a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16):
        raise TypeError(f"bf16 GEMM requires bf16 inputs, got {a.dtype}/{b.dtype}")
    return _mod().fp8_mm(a, b, sx, sw)


@fp8_mm.register_kernel("cpu")
def _fp8_mm_cpu(a, b, sx, sw):
    return torch.mm(a.float(), b.float().t()).to(torch.bfloat16)


@custom_op("custom::fp8_mm_prequant", mutates_args=())
def fp8_mm_prequant(
    a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Pre-quantized FP8 inputs, fused FP8 GEMM, FP32 accumulation, BF16 out."""


@fp8_mm_prequant.register_fake
def _fp8_mm_prequant_fake(a, b, scale):
    return torch.empty((a.size(0), b.size(0)), device=a.device, dtype=torch.bfloat16)


@fp8_mm_prequant.register_kernel("cuda")
def _fp8_mm_prequant_cuda(a, b, scale):
    if not (a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn):
        raise TypeError(
            f"pre-quantized FP8 GEMM requires fp8 inputs, got {a.dtype}/{b.dtype}"
        )
    return _mod().fp8_mm_prequant(a, b, scale)


@fp8_mm_prequant.register_kernel("cpu")
def _fp8_mm_prequant_cpu(a, b, scale):
    return (a.float() @ b.float().t() * scale).to(torch.bfloat16)


@custom_op("custom::fp8_mm_prequant_fp8", mutates_args=())
def fp8_mm_prequant_fp8(
    a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, out_scale: torch.Tensor
) -> torch.Tensor:
    """FP8 inputs and FP8 output: fused FP8 GEMM with FP32 accumulation."""


@fp8_mm_prequant_fp8.register_fake
def _fp8_mm_prequant_fp8_fake(a, b, scale, out_scale):
    return torch.empty((a.size(0), b.size(0)), device=a.device, dtype=a.dtype)


@fp8_mm_prequant_fp8.register_kernel("cuda")
def _fp8_mm_prequant_fp8_cuda(a, b, scale, out_scale):
    if not (a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn):
        raise TypeError(
            f"pre-quantized FP8 GEMM requires fp8 inputs, got {a.dtype}/{b.dtype}"
        )
    return _mod().fp8_mm_prequant_fp8(a, b, scale, out_scale)


@fp8_mm_prequant_fp8.register_kernel("cpu")
def _fp8_mm_prequant_fp8_cpu(a, b, scale, out_scale):
    return (a.float() @ b.float().t() * scale * out_scale).to(torch.float8_e4m3fn)


def linear_forward_scaled(x, w, bias, sx, sw, sx_inv, sw_inv, amax_x, amax_w):
    """Quantize BF16 inputs to FP8, accumulate in FP32, and return BF16.

    x/w: [..., K] / [N, K] bf16; sx/sw and their inverses control the fused
    E4M3 conversion; amax_x/amax_w receive the input max-abs values.
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
