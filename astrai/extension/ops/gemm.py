"""GEMM-kernel interface adapter (the only module touching the pybind).

One adapter per compiled module: this file covers the whole ``gemm``
binding surface, stateless and called directly — the fp8-symmetric entry
and the dtype-generic quantized family. The mma always consumes bf16
fragments; int8 operands dequantize in-register between the smem read and
the mma — never a separate F2F pass — and scales fold multiplicatively into
the epilogue.

- ``mm_fp8(a8, b8, scale) -> bf16`` — pre-quantized FP8 GEMM (symmetric fp8
  mma; ``scale`` is the combined dequant scale)
- ``mm_w8a16(a bf16, w int8, w_scale) -> bf16`` — weight-only quantized
  linear (``w`` is the int8 weight, ``w_scale`` per-channel ``[n]`` or a
  1-element per-tensor scalar)
- ``mm_w8a8(a int8, w int8, a_scale, w_scale) -> bf16`` — dynamic quantized
  GEMM (``a_scale`` per-row ``[m]`` or per-tensor)
- ``mm_w16a16(a bf16, b bf16) -> bf16`` — the family's plain-bf16 baseline

Scales are contiguous float32 CUDA tensors. The ``trans_a``/``trans_b``
flags name the math (``True`` = operand laid out ``[contract][rows]``);
inner-transposed views fold into the kernel layout at zero copy. ``bias``
(CUDA bf16 1D of length n) fuses into the epilogue.

Policy (fp8 recipes / autocast, int8 quantizers) lives in
``astrai.extension.quantize``; this module is stateless.
"""

from typing import Optional

import torch

from astrai.extension.loader import get_module


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
    return get_module("gemm").mm_fp8(a, b, scale, trans_a, trans_b, bias)


def mm_w8a16(
    a: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weight-only quantized linear: ``a @ (w * w_scale) (+ bias)``.

    ``a`` is bf16 (2D or 3D batched); ``w`` is the int8 weight in ``[N][K]``
    storage (``trans_b=True``, the nn.Linear convention). ``w_scale`` is a
    float32 CUDA tensor with either one element (per-tensor) or ``n``
    elements (per-output-channel). The result is bf16.
    """
    return get_module("gemm").mm_w8a16(a, w, w_scale, trans_a, trans_b, bias)


def mm_w8a8(
    a: torch.Tensor,
    w: torch.Tensor,
    a_scale: torch.Tensor,
    w_scale: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dynamic quantized GEMM: ``(a * a_scale) @ (w * w_scale) (+ bias)``.

    Both operands are int8; each side re-applies its dequant scale in the
    epilogue (``a_scale`` per-row ``[m]`` or per-tensor, ``w_scale``
    per-channel ``[n]`` or per-tensor). The result is bf16.
    """
    return get_module("gemm").mm_w8a8(a, w, a_scale, w_scale, trans_a, trans_b, bias)


def mm_w16a16(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Plain bf16 GEMM over the same kernel family (no dequant insert)."""
    return get_module("gemm").mm_w16a16(a, b, trans_a, trans_b, bias)


__all__ = ["mm_fp8", "mm_w16a16", "mm_w8a16", "mm_w8a8"]
