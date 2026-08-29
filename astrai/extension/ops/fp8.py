"""FP8 CUDA kernel interface adapter (the only module touching the pybind).

Attention-style thin wrappers: one Python entry per binding, called directly
— no torch.library dispatch layer. Optional arguments (``ring_state``,
``bias``) keep native Optional semantics at the pybind boundary, and
in-place buffer updates (the delayed-scaling ring fold, like attention's
KV-cache appends) happen on-stream without mutation declarations. CUDA-only:
non-CUDA or unsupported inputs raise from the binding's TORCH_CHECKs.

- ``quantize(x, scale, fmt, transposed=False) -> (x8|x8T, amax)`` — BF16/FP16/FP32
  → FP8 with fused amax (``transposed`` picks the orientation; arity is fixed)
- ``quantize_dual(x, scale, fmt) -> (x8, x8T, amax)`` — both orientations, one read
- ``mm_fp8(a8, b8, sa, sb) -> out`` — pre-quantized FP8 GEMM (BF16 output)

``scale`` is the quantization multiplier (device scalar); ``fmt`` is
``"e4m3"`` or ``"e5m2"``. ``amax`` values are *returned*, never passed as
output arguments.

Policy (scales, amax history, delayed scaling, autocast) lives in ``fp8.py``;
this module is stateless.
"""

from typing import Optional, Tuple

import torch

from astrai.extension.loader import get_module

# fmt string -> kernel int (0 = E4M3, 1 = E5M2)
_FMT_TO_INT = {"e4m3": 0, "e5m2": 1}


def _fmt_int(fmt: str) -> int:
    try:
        return _FMT_TO_INT[fmt]
    except KeyError:
        raise ValueError(f"unsupported fp8 format {fmt!r} (expected 'e4m3' or 'e5m2')")


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    fmt: str = "e4m3",
    transposed: bool = False,
    ring_state: Optional[torch.Tensor] = None,
    hist_idx: int = 0,
    fp8_max: float = 448.0,
    pow2_margin: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Float (bf16/fp16/fp32) -> FP8 quantize with fused amax.

    ``scale`` is the quantization multiplier (device scalar); ``fmt`` selects
    E4M3 or E5M2. ``amax`` is a fresh 1-element float32 tensor.
    ``transposed=True`` swaps ``x8`` for ``x8T``, the ``[cols][rows]``
    row-major transpose of the quantized input — the K-contiguous operand
    orientation NT GEMMs want — at the same 2-tuple arity.

    ``ring_state`` (a 1D float32 CUDA buffer laid out
    ``[hist n | scale | legacy | amax | done]``) switches on the in-kernel
    delayed-scaling fold: the kernel's last block folds the amax into
    ``hist[hist_idx]`` and publishes the next scale as
    ``max(hist) / fp8_max / pow2_margin`` — the returned ``amax`` is then the
    self-cleaned persistent slot (reads zero). None keeps the classic
    fresh-amax return.
    """
    return get_module("fp8_ops").quantize(
        x,
        scale,
        _fmt_int(fmt),
        transposed,
        ring_state,
        hist_idx,
        fp8_max,
        pow2_margin,
    )


def quantize_dual(
    x: torch.Tensor,
    scale: torch.Tensor,
    fmt: str = "e4m3",
    ring_state: Optional[torch.Tensor] = None,
    hist_idx: int = 0,
    fp8_max: float = 448.0,
    pow2_margin: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual-orientation quantize: one read of ``x`` produces both the
    row-major ``x8`` and its transposed ``x8T`` (plus ``amax``), for tensors
    consumed by GEMMs in both orientations (backward ``g``).

    ``ring_state`` switches on the in-kernel delayed-scaling fold exactly as
    in :func:`quantize`.
    """
    return get_module("fp8_ops").quantize_dual(
        x, scale, _fmt_int(fmt), ring_state, hist_idx, fp8_max, pow2_margin
    )


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
    return get_module("fp8_ops").mm_fp8(a, b, scale, trans_a, trans_b, bias)
