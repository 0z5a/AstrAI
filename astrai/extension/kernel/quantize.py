"""Quantize-kernel interface adapter (the only module touching the pybind).

Attention-style thin wrappers: one Python entry per binding of the compiled
``quantize`` module, called directly — no torch.library dispatch layer.
Optional arguments (``ring_state``) keep native Optional semantics at the
pybind boundary, and in-place buffer updates (the delayed-scaling ring fold,
like attention's KV-cache appends) happen on-stream without mutation
declarations. CUDA-only: non-CUDA or unsupported inputs raise from the
binding's TORCH_CHECKs.

- ``quantize(x, scale, fmt, transposed=False) -> (x8|x8T, amax)`` — BF16/FP16/FP32
  → FP8 with a fused amax (``transposed`` picks the orientation; arity is fixed)
- ``quantize_dual(x, scale, fmt) -> (x8, x8T, amax)`` — both orientations, one read

``scale`` is the quantization multiplier (device scalar); ``fmt`` is the fp8
output dtype (``torch.float8_e4m3fn`` / ``torch.float8_e5m2``) — the binding
validates and dispatches on it. ``amax`` is the delayed-scaling fold's
raw-domain amax of the round, or ``None`` with no ``ring_state``.

FP8 policy — scales, amax history, delayed scaling, the autocast region —
lives in ``astrai.extension.policy.quantization.autocast``; the INT8 inference primitives below are stateless torch operations.
"""

from typing import Optional, Tuple

import torch

from astrai.extension.runtime.loader import get_module


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    fmt: torch.dtype = torch.float8_e4m3fn,
    transposed: bool = False,
    ring_state: Optional[torch.Tensor] = None,
    hist_idx: int = 0,
    hist_len: Optional[int] = None,
    fp8_max: Optional[float] = None,
    pow2_margin: float = 1.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Float (bf16/fp16/fp32) -> FP8 quantize (scale-then-cast).

    ``scale`` is the quantization multiplier (device scalar); ``fmt`` is the
    fp8 output dtype (E4M3 or E5M2) — the binding validates and dispatches
    on it. ``transposed=True`` swaps ``x8`` for ``x8T``, the
    ``[cols][rows]`` row-major transpose of the quantized input — the
    K-contiguous operand orientation NT GEMMs want — at the same 2-tuple
    arity.

    ``ring_state`` (a 1D float32 CUDA buffer) switches on the in-kernel
    delayed-scaling fold: the kernel's last block folds the round's amax into
    ``hist[hist_idx]``, publishes the next scale — plus its correctly rounded
    reciprocal, the slot the next call reads as its multiplier — as
    ``max(hist) / fp8_max / pow2_margin``, and reports that round's amax.
    By default, ``fp8_max`` follows ``fmt`` (E4M3 or E5M2).
    A ring also needs ``hist_len``; the window is not recoverable from
    ``numel`` (the composed ring's trailing pair overshoots).

    No ``ring_state`` means a pure scale+cast: no fused-amax reduction and
    ``amax`` is ``None``. Callers that want the amax reduce the input
    themselves (dynamic scaling measures ``x.abs().amax()``).
    """
    return get_module("quantize").quantize(
        x,
        scale,
        fmt,
        transposed,
        ring_state,
        hist_idx,
        hist_len,
        fp8_max,
        pow2_margin,
    )


def quantize_dual(
    x: torch.Tensor,
    scale: torch.Tensor,
    fmt: torch.dtype = torch.float8_e4m3fn,
    transposed_fmt: Optional[torch.dtype] = None,
    ring_state: Optional[torch.Tensor] = None,
    hist_idx: int = 0,
    hist_len: Optional[int] = None,
    fp8_max: Optional[float] = None,
    pow2_margin: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Dual-orientation quantize: one read of ``x`` produces both the
    row-major ``x8`` and its transposed ``x8T`` (plus ``amax``), for tensors
    consumed by GEMMs in both orientations (the backward ``g``, and — under
    the hybrid training pair — the forward operands, whose transposed copies
    the backward GEMMs need in their own format).

    ``transposed_fmt`` casts the transposed side in a different fp8 format
    from the same read (``None`` = the same ``fmt``); the two conversions are
    elementwise, so a mixed pass is bit-identical to two single-format passes.

    ``ring_state`` switches on the in-kernel delayed-scaling fold exactly as
    in :func:`quantize` (``hist_len`` included); with no ``ring_state`` the
    kernel runs a pure scale+cast and ``amax`` is ``None``.
    """
    return get_module("quantize").quantize_dual(
        x,
        scale,
        fmt,
        transposed_fmt,
        ring_state,
        hist_idx,
        hist_len,
        fp8_max,
        pow2_margin,
    )


def __getattr__(name: str):
    # PEP 562: the binding (and with it the .so load) stays lazy — importing
    # this module on a box without the extension must keep working.
    if name == "K_FOLD_SLOTS":
        return get_module("quantize").K_FOLD_SLOTS
    raise AttributeError(name)


def quantize_weight_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel int8 quantization of a linear weight.

    ``w`` is ``[N, K]`` (the nn.Linear convention); returns
    ``(w8 int8 [N, K] contiguous, scale f32 [N])`` with
    ``w ≈ w8.float() * scale[:, None]`` and ``scale = amax(N) / 127``.
    Quantization runs in float32 regardless of the source dtype.
    """
    wf = w.detach().to(torch.float32)
    scale = wf.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round(wf / scale.unsqueeze(-1)).clamp_(-127, 127)
    return q.to(torch.int8).contiguous(), scale.contiguous()


def quantize_act_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row dynamic int8 quantization of activations.

    ``x`` is ``[..., K]``; returns ``(q int8 with x's shape, scale f32
    [prod(leading dims)])`` — upstream of ``quant_gemm``'s per-row a_scale
    (the scale contract: ``docs/developer/kernels/gemm.md``, "Scales").
    """
    xf = x.detach().to(torch.float32)
    x2 = xf.reshape(-1, xf.shape[-1])
    scale = x2.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp_(-127, 127)
    return q.to(torch.int8).reshape(x.shape), scale


__all__ = [
    "quantize",
    "quantize_dual",
    "quantize_weight_int8",
    "quantize_act_int8",
    "K_FOLD_SLOTS",
]
