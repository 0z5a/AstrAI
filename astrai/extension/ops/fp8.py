"""FP8 CUDA kernel interface adapter (the only module touching the pybind).

Isolates the ``fp8_ops`` CUDA extension behind stable Python primitives:

- ``quantize(x, scale, fmt) -> (x8, amax)`` — BF16/FP16/FP32 → FP8 with fused amax
- ``mm_fp8(a8, b8, sa, sb) -> out`` — pre-quantized FP8 GEMM (BF16 output)

Scale semantics: scales are *quantization steps* — the value divided out when
quantizing (``x8 = x / scale``). Every primitive computes its own inverse
internally; callers never pass ``scale_inv``. ``amax`` values are *returned*,
never passed as output arguments. ``fmt`` is ``"e4m3"`` or ``"e5m2"``.

Policy (scales, amax history, delayed scaling, autocast) lives in ``fp8.py``;
this module is stateless.
"""

from typing import Optional, Tuple

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


def _fmt_name(fmt: int) -> str:
    if fmt == 0:
        return "e4m3"
    if fmt == 1:
        return "e5m2"
    raise ValueError(f"unsupported quantization type {fmt!r}")


def _fmt_dtype(fmt: str) -> torch.dtype:
    return torch.float8_e5m2 if _fmt_int(fmt) else torch.float8_e4m3fn


@custom_op("custom::fp8_quantize", mutates_args=())
def fp8_quantize(
    x: torch.Tensor, scale: torch.Tensor, fmt: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Float (bf16/fp16/fp32) -> FP8 quantize with fused amax; ``scale`` is a multiplier."""


@fp8_quantize.register_fake
def _fp8_quantize_fake(x, scale, fmt):
    dtype = torch.float8_e5m2 if fmt == 1 else torch.float8_e4m3fn
    return (
        torch.empty(x.shape, device=x.device, dtype=dtype),
        torch.empty(1, device=x.device, dtype=torch.float32),
    )


_QUANT_INPUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@fp8_quantize.register_kernel("cuda")
def _fp8_quantize_cuda(x, scale, fmt):
    if x.dtype not in _QUANT_INPUT_DTYPES:
        raise TypeError(f"fp8 quantize requires bf16/fp16/fp32 input, got {x.dtype}")
    return get_module("fp8_ops").quantize(x, scale, int(fmt))


@fp8_quantize.register_kernel("cpu")
def _fp8_quantize_cpu(x, scale, fmt):
    x8 = (x.float() * scale).to(_fmt_dtype(_fmt_name(fmt)))
    amax = x.abs().amax().float().reshape(1).clamp_min(1e-12)
    return x8, amax


@custom_op("custom::fp8_gemm", mutates_args=())
def fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    trans_a: int = 0,
    trans_b: int = 0,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 GEMM: ``a @ b * scale (+ bias)`` with FP32 accumulation.

    2D or 3D (batched) operands; a size-1 batch broadcasts (matmul rules).
    ``bias`` (bf16, length n) fuses into the epilogue in fp32 before the
    single bf16 rounding. The result is always BF16; FP8 output is a
    separate quantize operation.
    """


@fp8_gemm.register_fake
def _fp8_gemm_fake(a, b, scale, trans_a=0, trans_b=0, bias=None):
    dtype = torch.bfloat16
    rows = a.size(2) if trans_a else a.size(1)
    cols = b.size(1) if trans_b else b.size(2)
    batches = [t.size(0) for t in (a, b) if t.dim() == 3]
    shape = (max(batches), rows, cols) if batches else (rows, cols)
    return torch.empty(shape, device=a.device, dtype=dtype)


@fp8_gemm.register_kernel("cuda")
def _fp8_gemm_cuda(a, b, scale, trans_a=0, trans_b=0, bias=None):
    if a.dtype != b.dtype or a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError(
            f"fp8 GEMM requires matching fp8 inputs, got {a.dtype}/{b.dtype}"
        )
    return get_module("fp8_ops").mm_fp8(a, b, scale, trans_a, trans_b, bias)


@fp8_gemm.register_kernel("cpu")
def _fp8_gemm_cpu(a, b, scale, trans_a=0, trans_b=0, bias=None):
    aa = a.float().transpose(-2, -1) if trans_a else a.float()
    bb = b.float().transpose(-2, -1) if trans_b else b.float()
    acc = aa @ bb * scale
    if bias is not None and bias.numel() > 0:
        acc = acc + bias.float()
    return acc.to(torch.bfloat16)


def quantize(
    x: torch.Tensor, scale: torch.Tensor, fmt: str = "e4m3"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Float (bf16/fp16/fp32) -> FP8 quantize with fused amax; returns
    ``(x8, amax)``.

    ``scale`` is the quantization multiplier (device scalar); ``fmt`` selects
    E4M3 or E5M2. ``amax`` is a fresh 1-element float32 tensor.
    """
    # Hot-path bypass of the torch.library dispatch (~5us/call, ~40% of a
    # 512-wide GEMM): real CUDA tensors of a supported dtype go straight to
    # the extension. Fake/subclass tensors and non-CUDA inputs keep the
    # custom_op route so torch.compile / meta / fake-tensor tracing and the
    # CPU fallback behave exactly as before.
    if (
        type(x) is torch.Tensor
        and x.is_cuda
        and x.dtype in _QUANT_INPUT_DTYPES
        and fmt in _FMT_TO_INT
    ):
        return get_module("fp8_ops").quantize(x, scale, _FMT_TO_INT[fmt])
    return fp8_quantize(x, scale, _fmt_int(fmt))


def mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pre-quantized FP8 GEMM: ``a @ b * scale (+ bias)``.

    ``a``/``b`` must be FP8 tensors of the same format, 2D or 3D (batched,
    matmul-style broadcast on the batch dim). Inner-transposed views (e.g.
    ``x.t()``) fold into the layout at zero copy. ``scale`` is their combined
    dequantization scale. ``bias`` (CUDA bf16 1D of length n) adds inside the
    kernel epilogue in fp32 — no separate elementwise pass. The result is
    BF16; FP8 output is a separate quantize operation.
    """
    # Same hot-path bypass as quantize(): the binding's TORCH_CHECKs keep
    # validation identical on the direct route (bias may be None — the
    # binding resolves it to the no-bias path).
    if (
        type(a) is torch.Tensor
        and a.is_cuda
        and a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    ):
        return get_module("fp8_ops").mm_fp8(
            a, b, scale, int(trans_a), int(trans_b), bias
        )
    return fp8_gemm(a, b, scale, trans_a, trans_b, bias)
