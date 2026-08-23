"""FP8 CUDA kernel interface adapter (the only module touching the pybind).

Isolates the ``fp8_ops`` CUDA extension behind stable Python primitives:

- ``quantize_bf16(x, scale, fmt) -> (x8, amax)`` — BF16 → FP8 with fused amax
- ``mm_fp8(a8, b8, sa, sb) -> out`` — pre-quantized FP8 GEMM (BF16 output)
- ``linear_forward_fp8(x, w, bias, sx, sw) -> (out, amax_x, amax_w)``
- ``linear_backward_fp8(g, x, w, masks, sg, sw, sx, fmt) -> (gx, gw, gb, amax_g)``

Scale semantics: scales are *quantization steps* — the value divided out when
quantizing (``x8 = x / scale``). Every primitive computes its own inverse
internally; callers never pass ``scale_inv``. ``amax`` values are *returned*,
never passed as output arguments. ``fmt`` is ``"e4m3"`` or ``"e5m2"``.

Policy (scales, amax history, delayed scaling, autocast) lives in ``fp8.py``;
this module is stateless.
"""

import torch
from torch.library import custom_op

from astrai.extension.loader import get_module

# fmt string -> kernel int (0 = E4M3, 1 = E5M2)
_FMT_TO_INT = {"e4m3": 0, "e5m2": 1}


def _fmt_int(fmt: str) -> int:
    try:
        return _FMT_TO_INT[fmt]
    except KeyError:
        raise ValueError(f"unsupported fp8 format {fmt!r} (expected 'e4m3' or 'e5m2')")


def _fmt_dtype(fmt: str) -> torch.dtype:
    return torch.float8_e5m2 if _fmt_int(fmt) else torch.float8_e4m3fn


@custom_op("custom::fp8_quantize", mutates_args=())
def fp8_quantize(
    x: torch.Tensor, scale: torch.Tensor, fmt: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 -> FP8 quantize with fused amax; returns ``(x8, amax)``."""


@fp8_quantize.register_fake
def _fp8_quantize_fake(x, scale, fmt):
    dtype = torch.float8_e5m2 if fmt else torch.float8_e4m3fn
    return (
        torch.empty(x.shape, device=x.device, dtype=dtype),
        torch.empty(1, device=x.device, dtype=torch.float32),
    )


@fp8_quantize.register_kernel("cuda")
def _fp8_quantize_cuda(x, scale, fmt):
    if x.dtype != torch.bfloat16:
        raise TypeError(f"fp8 quantize requires bf16 input, got {x.dtype}")
    return get_module("fp8_ops").quantize_bf16(x, scale, int(fmt))


@fp8_quantize.register_kernel("cpu")
def _fp8_quantize_cpu(x, scale, fmt):
    x8 = (x.float() / scale).to(_fmt_dtype("e5m2" if fmt else "e4m3"))
    amax = x.abs().amax().float().reshape(1).clamp_min(1e-12)
    return x8, amax


@custom_op("custom::fp8_gemm", mutates_args=())
def fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: int = 0,
    out_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 GEMM: ``a @ b^T * (sa * sb)`` with FP32 accumulation.

    ``out_dtype``: 0 = BF16 (default), 1 = FP8 E4M3 (requires ``out_scale``,
    the quantization step for the output — mirrors ``torch._scaled_mm``).
    """


@fp8_gemm.register_fake
def _fp8_gemm_fake(a, b, sa, sb, out_dtype=0, out_scale=None):
    dtype = torch.float8_e4m3fn if out_dtype else torch.bfloat16
    return torch.empty((a.size(0), b.size(0)), device=a.device, dtype=dtype)


@fp8_gemm.register_kernel("cuda")
def _fp8_gemm_cuda(a, b, sa, sb, out_dtype=0, out_scale=None):
    if a.dtype != b.dtype or a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError(
            f"fp8 GEMM requires matching fp8 inputs, got {a.dtype}/{b.dtype}"
        )
    return get_module("fp8_ops").mm_fp8(a, b, sa, sb, int(out_dtype), out_scale)


@fp8_gemm.register_kernel("cpu")
def _fp8_gemm_cpu(a, b, sa, sb, out_dtype=0, out_scale=None):
    acc = a.float() @ b.float().t() * sa * sb
    if out_dtype:
        os_ = 1.0 if out_scale is None else out_scale
        return (acc * os_).to(torch.float8_e4m3fn)
    return acc.to(torch.bfloat16)


def quantize_bf16(x: torch.Tensor, scale: torch.Tensor, fmt: str = "e4m3"):
    """BF16 -> FP8 quantize with fused amax; returns ``(x8, amax)``.

    ``scale`` is the quantization step (device scalar); ``fmt`` selects
    E4M3 or E5M2. ``amax`` is a fresh 1-element float32 tensor — the caller
    never clears it.
    """
    return fp8_quantize(x, scale, _fmt_int(fmt))


def mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: str = "bf16",
    out_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pre-quantized FP8 GEMM: ``a @ b^T * (sa * sb)``.

    ``a``/``b`` must be FP8 tensors of the same format (E4M3 or E5M2);
    ``sa``/``sb`` are their quantization steps. ``out_dtype`` is ``"bf16"``
    (default) or ``"e4m3"`` — FP8 output for layer-to-layer pipelines, which
    requires ``out_scale`` (the output quantization step).
    """
    if out_dtype not in ("bf16", "e4m3"):
        raise ValueError(
            f"unsupported out_dtype {out_dtype!r} (expected 'bf16' or 'e4m3')"
        )
    return fp8_gemm(a, b, sa, sb, int(out_dtype == "e4m3"), out_scale)


def linear_forward_fp8(x, w, bias, sx, sw, fmt: str = "e4m3"):
    """Pure FP8 linear forward: quantize x/w to ``fmt``, pre-quantized GEMM.

    Returns ``(out, amax_x, amax_w)``. ``bias`` may be ``None``. Both
    operands share the same FP8 format (E4M3 by default; E5M2 for a
    range-first configuration).
    """
    if not (x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16):
        raise TypeError(f"fp8 forward requires bf16 inputs, got {x.dtype}/{w.dtype}")
    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)
    return get_module("fp8_ops").linear_forward_fp8(x, w, bias, sx, sw, _fmt_int(fmt))


def linear_backward_fp8(g, x, w, masks, sg, sw, sx, fmt: str = "e5m2"):
    """FP8 linear backward; returns ``(grad_input, grad_weight, grad_bias, amax_g)``.

    The gradient (and the transposed w/x operands) are quantized to ``fmt``
    (default E5M2 — larger dynamic range for gradients) and the two GEMMs run
    as FP8 tensor-core products sharing a single gradient quantization.
    """
    if not (
        g.dtype == torch.bfloat16
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
    ):
        raise TypeError(
            f"fp8 backward requires bf16 inputs, got {g.dtype}/{x.dtype}/{w.dtype}"
        )
    return get_module("fp8_ops").linear_backward_fp8(
        g, x, w, list(masks), sg, sw, sx, _fmt_int(fmt)
    )
