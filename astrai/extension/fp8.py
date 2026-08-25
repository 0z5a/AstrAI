"""FP8 training: scaling recipes, per-tensor state, and aten::linear dispatch.

Layered (see also ``ops/fp8.py`` for the CUDA interface adapter):

1. Kernel interface: ``ops.fp8`` — the only module touching the pybind.
2. Training state (this module): scaling *recipes* (TE-style delayed scaling
   or dynamic current-amax scaling), per-tensor scales + amax history, and
   the ``fp8_autocast`` context (like ``torch.autocast``).
3. aten::linear integration (this module): registers the CUDA impl and the
   dtype guard.

Usage::

    from astrai.extension.fp8 import fp8_autocast

    with fp8_autocast(enabled=True, fp8_format="hybrid"):
        logits = model(input_ids)
    loss.backward()  # fp8 backward runs wherever it is called: the
    # forward captures the fmt/recipe/meta on the autograd node

Importing this module registers the aten::linear CUDA and AutogradCUDA
implementations.

Format defaults follow the ecosystem consensus: E4M3 for the forward pass,
E5M2 for the backward (gradient) pass ("hybrid"); every operand's scale is a
quantization step derived from its amax history by the active recipe.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch.library import Library

from astrai.extension.ops.fp8 import (
    linear_backward_fp8,
    linear_forward_fp8,
)

# Max representable value per FP8 format (E4M3: 448, E5M2: 57344).
FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}


class FP8Format(str, Enum):
    """Per-direction FP8 format. HYBRID = E4M3 forward / E5M2 backward."""

    E4M3 = "e4m3"
    E5M2 = "e5m2"
    HYBRID = "hybrid"

    def fwd(self) -> str:
        return "e4m3" if self is FP8Format.HYBRID else self.value

    def bwd(self) -> str:
        return "e5m2" if self is FP8Format.HYBRID else self.value


class FP8Recipe:
    """Scale-from-amax policy; the scale computation is the injection point.

    ``scale_from_history`` receives the amax tensor for this operand (a ring
    window for delayed scaling, the current amax for dynamic scaling) and
    returns the quantization step: ``scale = (amax / FP8_MAX[fmt]) / 2^margin``.
    """

    history_len: int = 16
    margin: int = 0

    def scale_from_history(self, amax: torch.Tensor, fmt: str) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class DelayedScaling(FP8Recipe):
    """TE-style delayed scaling: max over the amax history window.

    The scale is computed from amax measured in *previous* steps (delayed one
    step); the window length trades responsiveness against stability.
    """

    history_len: int = 16
    margin: int = 0

    def scale_from_history(self, amax: torch.Tensor, fmt: str) -> torch.Tensor:
        peak = amax.max()
        return ((peak / FP8_MAX[fmt]) / (2**self.margin)).clamp_min(1e-12)


@dataclass
class DynamicScaling(FP8Recipe):
    """Current-amax scaling (torchao DYNAMIC): measure, then quantize.

    No history — the scale is derived from the amax of the tensor being
    quantized in the same step, at the cost of an extra reduction pass.
    """

    history_len: int = 1
    margin: int = 0

    def scale_from_history(self, amax: torch.Tensor, fmt: str) -> torch.Tensor:
        peak = amax.max()
        return ((peak / FP8_MAX[fmt]) / (2**self.margin)).clamp_min(1e-12)


class _ScaleRing:
    """One operand's delayed-scaling state, packed for in-kernel finalization.

    ``state`` is a single float32 CUDA buffer ``[hist[n] | scale | counter]``
    (``hist`` / ``scale`` are views). The quantize kernel's last-finishing
    block records the freshly measured amax into ``hist[idx]``, reduces the
    window and publishes the next step's scale entirely on device — the
    Python-side hist-write / max / scale-write chain is gone. The counter
    slot stays int32-zero (float bits) between launches. ``idx`` advances
    host-side each step; ``margin`` is fixed by the recipe.
    """

    __slots__ = ("recipe", "state", "hist", "scale", "idx", "initialized")

    def __init__(self, device: torch.device, recipe: FP8Recipe):
        self.recipe = recipe
        n = recipe.history_len
        # [hist | scale | counter]; the counter slot must start at int 0.
        self.state = torch.zeros(n + 2, device=device, dtype=torch.float32)
        self.hist = self.state[:n]
        self.scale = self.state[n : n + 1]
        self.idx = 0
        self.initialized = False

    def advance(self) -> None:
        """Rotate to the next history slot after an in-kernel finalize."""
        self.idx = (self.idx + 1) % self.hist.numel()

    def seed(self, t: torch.Tensor, fmt: str) -> None:
        amax = t.abs().amax().to(torch.float32).clamp_min(1e-12)
        self.hist.fill_(amax)
        self.scale.copy_(self.recipe.scale_from_history(self.hist, fmt))
        self.initialized = True


class FP8TensorMeta:
    """Per-weight delayed-scaling state: one ring per operand role.

    Holds the ``w`` / ``x`` / ``g`` rings; fused kernels record the amax
    while quantizing, so the scale used at step N reflects amax from steps
    < N. DynamicScaling never allocates a meta — it measures the current
    amax inline (``_dynamic_scale``), so it needs no history storage.
    """

    __slots__ = ("w", "x", "g")

    def __init__(self, device: torch.device, recipe: FP8Recipe):
        self.w = _ScaleRing(device, recipe)
        self.x = _ScaleRing(device, recipe)
        self.g = _ScaleRing(device, recipe)


class FP8State:
    """Global fp8 training state: active recipe + per-tensor metas."""

    def __init__(self):
        self.enabled = False
        self.recipe: FP8Recipe = DelayedScaling()
        self.fp8_format: FP8Format = FP8Format.HYBRID
        self._metas: dict[tuple, FP8TensorMeta] = {}

    def get_weight_meta(self, w: torch.Tensor) -> FP8TensorMeta:
        key = (w.data_ptr(), w.shape, w.dtype)
        meta = self._metas.get(key)
        if meta is None:
            meta = FP8TensorMeta(w.device, self.recipe)
            self._metas[key] = meta
        return meta

    def reset(self) -> None:
        self.enabled = False
        self._metas.clear()


# Global singleton: autograd backward runs on the engine worker threads, so
# thread-local state would lose the fp8 flag during loss.backward(). The GIL
# protects Python-side mutation; the CUDA kernels take their own mutex.
_state = FP8State()


def fp8_state() -> FP8State:
    return _state


@contextmanager
def fp8_autocast(
    enabled: bool = True,
    update_interval: int = 16,
    recipe: Optional[FP8Recipe] = None,
    fp8_format: str = "hybrid",
    margin: int = 0,
):
    """Autocast-style context: fp8 linear dispatch on this thread.

    Usage::

        with fp8_autocast(enabled=True, fp8_format="hybrid"):
            logits = model(input_ids)   # aten::linear -> fp8 path
        loss.backward()  # fp8 backward; state was captured at forward time

    Args:
        enabled: toggle fp8 dispatch for aten::linear.
        update_interval: legacy alias for the delayed-scaling history window
            (used only when ``recipe`` is not given).
        recipe: scaling policy; defaults to ``DelayedScaling(update_interval)``.
        fp8_format: ``"e4m3"`` / ``"e5m2"`` / ``"hybrid"`` (default) — hybrid
            means E4M3 forward, E5M2 backward.
        margin: scale headroom (``scale = (amax / FP8_MAX) / 2^margin``) used
            with the default delayed recipe.
    """
    state = fp8_state()
    prev = (state.enabled, state.recipe, state.fp8_format)
    if recipe is None:
        recipe = DelayedScaling(history_len=update_interval, margin=margin)
    state.enabled = enabled
    state.recipe = recipe
    state.fp8_format = FP8Format(fp8_format)
    try:
        yield
    finally:
        state.enabled, state.recipe, state.fp8_format = prev


# ---------------------------------------------------------------------------
# Strategy-level forward / backward (called from the aten::linear impl)
# ---------------------------------------------------------------------------


def _dynamic_scale(t: torch.Tensor, recipe: FP8Recipe, fmt: str) -> torch.Tensor:
    amax = t.abs().amax().to(torch.float32).clamp_min(1e-12)
    return recipe.scale_from_history(amax, fmt)


def fp8_linear_forward(x: torch.Tensor, w: torch.Tensor, bias=None):
    """Scaled fp8 linear forward (called from the aten::linear impl).

    Pure FP8 path for both recipes: quantize x/w with the active scales, run
    the pre-quantized GEMM. With delayed scaling the rings finalize inside
    the quantize kernels (amax folded into the window, next step's scale
    published on device); dynamic scaling measures the current amax itself.
    """
    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)
    state = fp8_state()
    fmt = state.fp8_format.fwd()
    if isinstance(state.recipe, DynamicScaling):
        meta = None
        sx = _dynamic_scale(x.reshape(-1, w.size(1)), state.recipe, fmt)
        sw = _dynamic_scale(w, state.recipe, fmt)
        out, amax_x, amax_w = linear_forward_fp8(x, w, bias, sx, sw, fmt)
    else:
        meta = state.get_weight_meta(w)
        if not meta.w.initialized:
            meta.w.seed(w, fmt)
        if not meta.x.initialized:
            meta.x.seed(x, fmt)
        # In-kernel ring finalization: the kernels write hist[idx] and the
        # next scale; idx rotates host-side (the device counter self-rearms).
        w_is_fp8 = w.dtype != torch.bfloat16
        out, amax_x, amax_w = linear_forward_fp8(
            x,
            w,
            bias,
            meta.x.scale,
            meta.w.scale,
            fmt,
            None,
            meta.x.state,
            meta.x.idx,
            state.recipe.margin,
            None if w_is_fp8 else meta.w.state,
            meta.w.idx,
            state.recipe.margin,
        )
        meta.x.advance()
        if not w_is_fp8:
            meta.w.advance()
    return out


class _LinearFp8(torch.autograd.Function):
    """The fp8 linear forward/backward pair (standard Function style).

    The forward runs inside the ``fp8_autocast`` region and captures the
    active fmt/recipe/meta on ``ctx``; the backward reads only the captured
    state, so ``loss.backward()`` may run after the context exits. The
    gradient is quantized once (E5M2 in hybrid mode) and the dX / dW GEMMs
    share that quantization; output masks come from ``needs_input_grad``.
    """

    @staticmethod
    def forward(ctx, x, w, bias):
        out = fp8_linear_forward(x, w, bias)
        state = fp8_state()
        ctx.save_for_backward(x, w)
        ctx.fmt_bwd = state.fp8_format.bwd()
        ctx.recipe = state.recipe
        ctx.is_dynamic = isinstance(state.recipe, DynamicScaling)
        ctx.meta = None if ctx.is_dynamic else state.get_weight_meta(w)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        fmt = ctx.fmt_bwd
        if ctx.is_dynamic:
            sg = _dynamic_scale(g, ctx.recipe, fmt)
            sw = _dynamic_scale(w, ctx.recipe, fmt)
            sx = _dynamic_scale(x, ctx.recipe, fmt)
            grad_x, grad_w, grad_b, amax_g = linear_backward_fp8(
                g, x, w, list(ctx.needs_input_grad), sg, sw, sx, fmt
            )
        else:
            meta = ctx.meta
            if not meta.g.initialized:
                meta.g.seed(g, fmt)
            # The g quantize kernel finalizes the gradient's ring in-kernel.
            grad_x, grad_w, grad_b, amax_g = linear_backward_fp8(
                g,
                x,
                w,
                list(ctx.needs_input_grad),
                meta.g.scale,
                meta.w.scale,
                meta.x.scale,
                fmt,
                meta.g.state,
                meta.g.idx,
                ctx.recipe.margin,
            )
            meta.g.advance()
        return grad_x, grad_w, grad_b if ctx.needs_input_grad[2] else None


# ---------------------------------------------------------------------------
# aten::linear integration
# ---------------------------------------------------------------------------


def fp8_linear_enable(enabled: bool = True) -> None:
    """Toggle fp8 dispatch for aten::linear (global; backward runs on engine
    worker threads, so a thread-local flag would be lost during backward)."""
    fp8_state().enabled = enabled


def fp8_linear_enabled() -> bool:
    return fp8_state().enabled


def _fp8_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Shape guard for the fp8 linear path.

    Unlike a strict 16-alignment requirement, the fp8 kernels handle unaligned
    M/N via boundary checks (slower but correct) — so no whole-call bf16
    fallback for small decode batches. Only the K-dimension contraction must
    match, and the weight must be 2D.
    """
    return x.dim() >= 2 and w.dim() == 2 and x.size(-1) == w.size(1)


def _linear_cuda_impl(x: torch.Tensor, w: torch.Tensor, bias=None):
    if (
        fp8_linear_enabled()
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and _fp8_supported(x, w)
    ):
        return _LinearFp8.apply(x, w, bias)
    return torch.ops.aten.linear.default.redispatch(
        torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd),
        x,
        w,
        bias,
    )


_lib = Library("aten", "IMPL", "CUDA")
_lib.impl("linear", _linear_cuda_impl)
# Also replace torch's generated linear autograd formula (which would call
# aten::linear_backward after the fp8_autocast region exits). The fp8
# backward is owned by _LinearFp8 with its state captured at forward time,
# so loss.backward() works wherever it is called; the same CUDA registration
# still covers inference_mode, where autograd keys are skipped entirely.
_lib_autograd = Library("aten", "IMPL", "AutogradCUDA")
_lib_autograd.impl("linear", _linear_cuda_impl)
