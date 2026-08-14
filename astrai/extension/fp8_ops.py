"""FP8 matrix-multiply op (torch.library custom_op) and FP8 linear replacement.

Dispatch table:
- Meta (register_fake): shapes only, for torch.compile / dynamic shapes
- CUDA: csrc fp8_mm kernel (cuBLASLt TN fp8 GEMM, e4m3 in, fp32 acc/out)
- CPU: fp32 fallback (testing)
- AutogradCUDA (register_autograd): bf16 backward, scale-corrected
"""

import torch
from torch.library import custom_op

from astrai.extension.loader import get_module, is_available


@custom_op("custom::fp8_mm", mutates_args=())
def fp8_mm(
    a: torch.Tensor, b: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor
) -> torch.Tensor:
    """FP8 e4m3 GEMM: a[M,K] x b[N,K] -> fp32[M,N], scales applied by the caller.

    a/b arrive pre-scaled (divided by sx/sw) fp8 tensors; the op returns the
    unscaled fp32 result so scale math stays in autograd-land.
    """


@fp8_mm.register_fake
def _fp8_mm_fake(a, b, sx, sw):
    return torch.empty((a.size(0), b.size(1)), device=a.device, dtype=torch.float32)


@fp8_mm.register_kernel("cuda")
def _fp8_mm_cuda(a, b, sx, sw):
    if not is_available("fp8_mm"):
        raise RuntimeError(
            "CUDA kernel 'fp8_mm' is not available. Build with CSRC_KERNELS=true."
        )
    return get_module("fp8_mm").fp8_mm(a, b)


@fp8_mm.register_kernel("cpu")
def _fp8_mm_cpu(a, b, sx, sw):
    return torch.mm(a.float(), b.float().t())


def _fp8_mm_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _fp8_mm_backward(ctx, g):
    """Scale-corrected straight-through gradients.

    out = F(a, b) * (sx * sw) with F(a, b) = a @ b^T, a = x/sx, b = w/sw:
      dx = g * sw @ b          (dout/dx = dF/da * 1/sx * sx*sw)
      dW = (g * sx)^T @ a      (dout/dw = dF/db * 1/sw * sx*sw)
    bf16 GEMMs keep gradients in range (e4m3 saturates at 448).
    """
    a, b, sx, sw = ctx.saved_tensors
    ga = torch.mm(g * sw, b.float())
    gb = torch.mm((g * sx).t(), a.float())
    return ga.to(torch.bfloat16), gb.to(torch.bfloat16), None, None


fp8_mm.register_autograd(_fp8_mm_backward, setup_context=_fp8_mm_setup_context)


def fp8_linear_forward(x: torch.Tensor, w: torch.Tensor, bias=None):
    """TE-style scaled fp8 linear forward (delegates to fp8_state)."""
    from astrai.extension.fp8_state import fp8_linear_forward as _f

    return _f(x, w, bias)


def fp8_linear_backward(g, x, w, masks):
    """TE-style scaled fp8 linear backward (delegates to fp8_state)."""
    from astrai.extension.fp8_state import fp8_linear_backward as _b

    return _b(g, x, w, masks)


def fp8_available() -> bool:
    return is_available("fp8_mm")
