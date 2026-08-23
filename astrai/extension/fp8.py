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
        loss.backward()

Importing this module registers the aten::linear CUDA implementation.

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
E4M3_MAX = FP8_MAX["e4m3"]  # legacy alias


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


class FP8TensorMeta:
    """Per-tensor scaling state: amax history rings + derived scales.

    One ring per operand (weight / activation / gradient). Scales are derived
    from the ring by the recipe; fused kernels record the amax while
    quantizing, so the scale used at step N reflects amax from steps < N
    (delayed one step).
    """

    __slots__ = (
        "recipe",
        "w_hist",
        "x_hist",
        "g_hist",
        "w_idx",
        "x_idx",
        "g_idx",
        "w_scale",
        "x_scale",
        "g_scale",
        "w_init",
        "x_init",
        "g_init",
    )

    def __init__(self, device: torch.device, recipe: FP8Recipe):
        self.recipe = recipe
        n = recipe.history_len
        self.w_hist = torch.ones(n, device=device, dtype=torch.float32)
        self.x_hist = torch.ones(n, device=device, dtype=torch.float32)
        self.g_hist = torch.ones(n, device=device, dtype=torch.float32)
        self.w_idx = self.x_idx = self.g_idx = 0
        self.w_scale = torch.ones(1, device=device, dtype=torch.float32)
        self.x_scale = torch.ones(1, device=device, dtype=torch.float32)
        self.g_scale = torch.ones(1, device=device, dtype=torch.float32)
        self.w_init = self.x_init = self.g_init = False

    # -- ring helpers -------------------------------------------------------

    def _record(self, hist: torch.Tensor, idx: int, amax: torch.Tensor) -> int:
        hist[idx] = amax.reshape(())
        return (idx + 1) % hist.numel()

    def _refresh(self, hist: torch.Tensor, scale: torch.Tensor, fmt: str) -> None:
        scale.copy_(self.recipe.scale_from_history(hist, fmt))

    def _seed(
        self, hist: torch.Tensor, scale: torch.Tensor, t: torch.Tensor, fmt: str
    ) -> None:
        amax = t.abs().amax().to(torch.float32).clamp_min(1e-12)
        hist.fill_(amax)
        scale.copy_(self.recipe.scale_from_history(hist, fmt))

    # -- per-operand updates (delayed: record now, refresh for next step) ---

    def update_w(self, amax: torch.Tensor, fmt: str) -> None:
        self.w_idx = self._record(self.w_hist, self.w_idx, amax)
        self._refresh(self.w_hist, self.w_scale, fmt)

    def update_x(self, amax: torch.Tensor, fmt: str) -> None:
        self.x_idx = self._record(self.x_hist, self.x_idx, amax)
        self._refresh(self.x_hist, self.x_scale, fmt)

    def update_g(self, amax: torch.Tensor, fmt: str) -> None:
        self.g_idx = self._record(self.g_hist, self.g_idx, amax)
        self._refresh(self.g_hist, self.g_scale, fmt)

    # -- first-use seeding --------------------------------------------------

    def init_w(self, w: torch.Tensor, fmt: str) -> None:
        self._seed(self.w_hist, self.w_scale, w, fmt)
        self.w_init = True

    def init_x(self, x: torch.Tensor, fmt: str) -> None:
        self._seed(self.x_hist, self.x_scale, x, fmt)
        self.x_init = True

    def init_g(self, g: torch.Tensor, fmt: str) -> None:
        self._seed(self.g_hist, self.g_scale, g, fmt)
        self.g_init = True


class FP8State:
    """Global fp8 training state: active recipe + per-tensor metas."""

    def __init__(self):
        self.enabled = False
        self.recipe: FP8Recipe = DelayedScaling()
        self.fp8_format: FP8Format = FP8Format.HYBRID
        self._metas: dict[tuple, FP8TensorMeta] = {}
        self._last_device: Optional[torch.device] = None

    def get_weight_meta(self, w: torch.Tensor) -> FP8TensorMeta:
        key = (w.data_ptr(), w.shape, w.dtype)
        meta = self._metas.get(key)
        if meta is None:
            if self._last_device is None:
                self._last_device = w.device
            meta = FP8TensorMeta(w.device, self.recipe)
            self._metas[key] = meta
        return meta

    def reset(self) -> None:
        self.enabled = False
        self._metas.clear()
        self._last_device = None


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
            loss.backward()

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
    the pre-quantized GEMM, and feed the freshly measured amax back into the
    delayed-scaling ring (dynamic scaling measures the current amax itself).
    """
    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)
    state = fp8_state()
    fmt = state.fp8_format.fwd()
    meta = state.get_weight_meta(w)
    if not meta.w_init:
        meta.init_w(w, fmt)
    if isinstance(state.recipe, DynamicScaling):
        sx = _dynamic_scale(x.reshape(-1, w.size(1)), state.recipe, fmt)
        sw = _dynamic_scale(w, state.recipe, fmt)
    else:
        if not meta.x_init:
            meta.init_x(x, fmt)
        sx, sw = meta.x_scale, meta.w_scale
    out, amax_x, amax_w = linear_forward_fp8(x, w, bias, sx, sw, fmt)
    if not isinstance(state.recipe, DynamicScaling):
        meta.update_x(amax_x, fmt)
        meta.update_w(amax_w, fmt)
    return out


def fp8_linear_backward(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, masks):
    """Scaled fp8 linear backward (called from aten::linear_backward).

    The gradient is quantized to the backward format (E5M2 in hybrid mode)
    and the dX / dW GEMMs share that single quantization.
    """
    state = fp8_state()
    fmt = state.fp8_format.bwd()
    meta = state.get_weight_meta(w)
    if isinstance(state.recipe, DynamicScaling):
        sg = _dynamic_scale(g, state.recipe, fmt)
        sw = _dynamic_scale(w, state.recipe, fmt)
        sx = _dynamic_scale(x, state.recipe, fmt)
    else:
        if not meta.g_init:
            meta.init_g(g, fmt)
        sg, sw, sx = meta.g_scale, meta.w_scale, meta.x_scale
    grad_x, grad_w, grad_b, amax_g = linear_backward_fp8(
        g, x, w, masks, sg, sw, sx, fmt
    )
    if not isinstance(state.recipe, DynamicScaling):
        meta.update_g(amax_g, fmt)
    return grad_x, grad_w, grad_b


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
        return fp8_linear_forward(x, w, bias)
    return torch.ops.aten.linear.default.redispatch(
        torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd),
        x,
        w,
        bias,
    )


def _linear_backward_cuda_impl(input_tensor, grad_output, weight, output_mask):
    # Backward dim contract: grad_output is [..., N], weight is [N, K], so
    # the contraction check is grad_output.size(-1) == weight.size(0) (not the
    # forward's x.size(-1) == w.size(1) — that would silently skip fp8 for
    # every non-square layer).
    if (
        fp8_linear_enabled()
        and weight.dtype == torch.bfloat16
        and grad_output.dim() >= 2
        and weight.dim() == 2
        and grad_output.size(-1) == weight.size(0)
        and input_tensor.dim() >= 2
        and input_tensor.size(-1) == weight.size(1)
    ):
        return fp8_linear_backward(grad_output, input_tensor, weight, list(output_mask))
    compute_dtype = weight.dtype
    grad = grad_output.to(compute_dtype)
    grad_2d = grad.reshape(-1, weight.size(0))
    input_2d = input_tensor.reshape(-1, input_tensor.size(-1)).to(compute_dtype)
    # Unneeded grads come back full-shape-but-uninitialized (mirroring the
    # fp8 binding), so reshape_as can never hit an empty tensor.
    grad_input = (
        torch.mm(grad_2d, weight).reshape_as(input_tensor)
        if output_mask[0]
        else torch.empty_like(input_tensor)
    )
    grad_weight = (
        torch.mm(grad_2d.t(), input_2d) if output_mask[1] else torch.empty_like(weight)
    )
    grad_bias = (
        grad.sum(dim=0)
        if output_mask[2]
        else torch.empty(0, device=grad.device, dtype=grad.dtype)
    )
    return grad_input, grad_weight, grad_bias


_lib = Library("aten", "IMPL", "CUDA")
_lib.impl("linear", _linear_cuda_impl)
_lib.impl("linear_backward", _linear_backward_cuda_impl)
