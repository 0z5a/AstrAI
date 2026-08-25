"""FP8 training: scaling recipes, per-tensor state, and aten::linear dispatch.

Layered (see ``ops/fp8.py`` for the CUDA interface adapter):
1. ``ops.fp8`` — the only module touching the pybind.
2. This module (strategy layer): scaling *recipes* (TE-style delayed scaling
   or dynamic current-amax scaling), per-tensor scales + amax history, and the
   ``fp8_autocast`` context manager (like ``torch.autocast``).
3. aten::linear integration: registers the CUDA + AutogradCUDA impls.

Usage::

    from astrai.extension.fp8 import fp8_autocast
    with fp8_autocast(enabled=True, fp8_format="hybrid"):
        logits = model(input_ids)
    loss.backward()  # fp8 backward runs anywhere; fwd captured state on the node

Format defaults follow the ecosystem consensus: E4M3 forward / E5M2 backward
("hybrid"); every operand's scale is a quantization step derived from its amax
history by the active recipe.

The context mirrors ``torch.autocast`` (``autocast_mode.py``): the active
``(enabled, recipe, fp8_format)`` triple is thread-local (a ``contextvars``
``ContextVar``, absent outside any region), and the manager is class-based and
reentrant with nested ``enabled=False`` disabling dispatch inside it. The module
targets *training*: every step quantizes x/w/g fresh (no weight-cast cache — the
optimizer bumps the weight version each step, so a torch-style cached_cast would
miss anyway), and the per-operand scales come from the delayed/dynamic recipe.
"""

import functools
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch
from torch.library import Library

from astrai.extension.ops.fp8 import mm_fp8, quantize

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
    """Scale-from-amax policy: ``scale = (amax / FP8_MAX[fmt]) / 2^margin``.

    ``scale_from_history`` receives the operand's amax tensor (a ring window for
    delayed scaling, the current amax for dynamic scaling) and returns the
    quantization step. Subclasses set ``history_len`` / ``margin``.
    """

    history_len: int = 16
    margin: int = 0

    def scale_from_history(self, amax: torch.Tensor, fmt: str) -> torch.Tensor:
        peak = amax.max()
        return ((peak / FP8_MAX[fmt]) / (2**self.margin)).clamp_min(1e-12)


@dataclass
class DelayedScaling(FP8Recipe):
    """TE-style delayed scaling: max over the amax history window (amax from
    *previous* steps; the window trades responsiveness against stability)."""

    history_len: int = 16
    margin: int = 0


@dataclass
class DynamicScaling(FP8Recipe):
    """Current-amax scaling (torchao DYNAMIC): measure, then quantize. No
    history — the scale is derived from the same-step amax, at an extra pass."""

    history_len: int = 1
    margin: int = 0


class _ScaleRing:
    """One operand's delayed-scaling state: a float32 buffer
    ``[hist[n] | scale | counter]`` (views). ``update`` folds the amax
    returned by the quantize primitive into ``hist[idx]`` and publishes the
    next scale from the window; ``idx`` advances host-side each step. The
    trailing slot is a legacy counter kept for state-buffer compatibility.
    """

    __slots__ = ("recipe", "state", "hist", "scale", "idx", "initialized")

    def __init__(self, device: torch.device, recipe: FP8Recipe):
        self.recipe = recipe
        n = recipe.history_len
        self.state = torch.zeros(n + 2, device=device, dtype=torch.float32)
        self.hist = self.state[:n]
        self.scale = self.state[n : n + 1]
        self.idx = 0
        self.initialized = False

    def advance(self) -> None:
        """Rotate to the next history slot after metadata update."""
        self.idx = (self.idx + 1) % self.hist.numel()

    def seed(self, t: torch.Tensor, fmt: str) -> None:
        amax = t.abs().amax().to(torch.float32).clamp_min(1e-12)
        self.hist.fill_(amax)
        self.scale.copy_(self.recipe.scale_from_history(self.hist, fmt))
        self.initialized = True

    def update(self, amax: torch.Tensor, fmt: str) -> None:
        self.hist[self.idx].copy_(amax.reshape(()))
        self.scale.copy_(self.recipe.scale_from_history(self.hist, fmt))


class FP8TensorMeta:
    """Per-weight delayed-scaling state for ``w``, ``x`` and ``g``.

    DynamicScaling never allocates a meta; it measures the current amax inline.
    """

    __slots__ = ("w", "x", "g")

    def __init__(self, device: torch.device, recipe: FP8Recipe):
        self.w = _ScaleRing(device, recipe)
        self.x = _ScaleRing(device, recipe)
        self.g = _ScaleRing(device, recipe)


@dataclass(frozen=True)
class _ActiveConfig:
    """The immutable (enabled, recipe, format) triple of one open region."""

    enabled: bool
    recipe: FP8Recipe
    fp8_format: FP8Format


# Thread-local active configuration (torch's autocast TLS analog): set by
# fp8_autocast on __enter__, absent outside any region. Autograd engine
# threads run backwards with their own empty context — fine, since backward
# only reads state captured on ctx at forward time.
_active_config: ContextVar[Optional[_ActiveConfig]] = ContextVar(
    "astrai_fp8_active_config", default=None
)


class FP8State:
    """Global fp8 training state: per-tensor metas + out-of-region defaults.

    The active ``(enabled, recipe, fp8_format)`` triple is a ``ContextVar`` set
    by ``fp8_autocast``. The properties below read that active config when a
    region is open and the global defaults otherwise; the setters (and
    ``fp8_linear_enable``) write the global defaults — the persistent switch
    applying outside any region. The metas registry is shared across threads
    (GIL-protected); fp8 backward runs on autograd engine threads and only
    touches metas captured on ``ctx`` at forward time.
    """

    def __init__(self):
        self.default_enabled = False
        self.default_recipe: FP8Recipe = DelayedScaling()
        self.default_format: FP8Format = FP8Format.HYBRID
        self._metas: Dict[tuple, FP8TensorMeta] = {}

    # Active-config views (region config if open, else the defaults).
    @property
    def enabled(self) -> bool:
        cfg = _active_config.get()
        return cfg.enabled if cfg is not None else self.default_enabled

    @property
    def recipe(self) -> FP8Recipe:
        cfg = _active_config.get()
        return cfg.recipe if cfg is not None else self.default_recipe

    @property
    def fp8_format(self) -> FP8Format:
        cfg = _active_config.get()
        return cfg.fp8_format if cfg is not None else self.default_format

    # Persistent (out-of-region) defaults.
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.default_enabled = bool(value)

    @recipe.setter
    def recipe(self, value: FP8Recipe) -> None:
        self.default_recipe = value

    @fp8_format.setter
    def fp8_format(self, value: FP8Format) -> None:
        self.default_format = FP8Format(value)

    def get_weight_meta(self, w: torch.Tensor) -> FP8TensorMeta:
        key = (w.data_ptr(), w.shape, w.dtype)
        meta = self._metas.get(key)
        if meta is None:
            meta = FP8TensorMeta(w.device, self.recipe)
            self._metas[key] = meta
        return meta

    def reset(self) -> None:
        """Restore construction defaults (switch, recipe, format) and drop all
        per-weight metas — a full state reset for tests / reconfiguration."""
        self.default_enabled = False
        self.default_recipe = DelayedScaling()
        self.default_format = FP8Format.HYBRID
        self._metas.clear()


# Process-wide singleton; per-thread/per-region state lives in _active_config.
_state = FP8State()


def fp8_state() -> FP8State:
    return _state


def _active() -> Optional[_ActiveConfig]:
    """The active config when fp8 dispatch is on, else ``None`` (fast guard).

    A region config wins (honoring nested ``enabled=False`` regions); with no
    region open this falls back to the persistent global switch
    (``fp8_linear_enable``), so that flag still routes aten::linear to fp8.
    """
    cfg = _active_config.get()
    if cfg is not None:
        return cfg if cfg.enabled else None
    if _state.default_enabled:
        return _ActiveConfig(True, _state.default_recipe, _state.default_format)
    return None


def _current_config() -> _ActiveConfig:
    """Like ``_active()`` but always returns a config (disabled regions and
    out-of-region direct calls resolve to the global defaults)."""
    cfg = _active_config.get()
    if cfg is not None:
        return cfg
    return _ActiveConfig(
        _state.default_enabled, _state.default_recipe, _state.default_format
    )


class fp8_autocast:
    """Autocast-style context: fp8 linear dispatch on this thread.

    Mirrors ``torch.autocast`` — a class-based, reentrant, nestable context
    over thread-local state::

        with fp8_autocast(enabled=True, fp8_format="hybrid"):
            logits = model(input_ids)   # aten::linear -> fp8 path
        loss.backward()  # fp8 backward; state was captured at forward time

    Nesting follows torch: each ``__enter__`` pushes the new active config, each
    ``__exit__`` restores the previous one, and a nested ``enabled=False`` region
    simply disables dispatch inside it. The instance doubles as a decorator.
    """

    def __init__(
        self,
        enabled: bool = True,
        update_interval: int = 16,
        recipe: Optional[FP8Recipe] = None,
        fp8_format: str = "hybrid",
        margin: int = 0,
    ):
        if recipe is None:
            recipe = DelayedScaling(history_len=update_interval, margin=margin)
        self._config = _ActiveConfig(bool(enabled), recipe, FP8Format(fp8_format))
        self._tokens: List[Token] = []

    def __enter__(self) -> "fp8_autocast":
        self._tokens.append(_active_config.set(self._config))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        token = self._tokens.pop()
        _active_config.reset(token)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate


# ---------------------------------------------------------------------------
# Strategy-level forward / backward (called from the aten::linear impl)
# ---------------------------------------------------------------------------


def _dynamic_scale(t: torch.Tensor, recipe: FP8Recipe, fmt: str) -> torch.Tensor:
    amax = t.abs().amax().to(torch.float32).clamp_min(1e-12)
    return recipe.scale_from_history(amax, fmt)


def _is_fp8(dtype: torch.dtype) -> bool:
    """A pre-quantized weight takes the GEMM directly (no re-quantize)."""
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


_zero_bias: Dict[Optional[int], torch.Tensor] = {}


def _empty_bias(x: torch.Tensor) -> torch.Tensor:
    """Per-device cached 0-element bf16 bias (the binding only checks numel —
    never mutated), saving a CUDA allocation per bias-less linear."""
    key = x.device.index
    t = _zero_bias.get(key)
    if t is None:
        t = torch.empty(0, device=x.device, dtype=torch.bfloat16)
        _zero_bias[key] = t
    return t


def fp8_linear_forward(
    x: torch.Tensor, w: torch.Tensor, bias=None, cfg: Optional[_ActiveConfig] = None
):
    """Scaled fp8 linear forward (called from the aten::linear impl).

    Composed from the two stateless primitives: quantize x/w with the active
    scales, run the pre-quantized GEMM, add the bias. Delayed scaling folds
    the returned amax into the history ring and publishes the next scale;
    dynamic scaling measures the current amax itself. Training quantizes the
    weight every step (the optimizer bumps its version, so there is no cast
    cache, matching ``cached_cast``-less behavior).
    """
    state = fp8_state()
    if cfg is None:
        cfg = _current_config()
    fmt = cfg.fp8_format.fwd()
    if bias is None:
        bias = _empty_bias(x)
    if isinstance(cfg.recipe, DynamicScaling):
        sx = _dynamic_scale(x.reshape(-1, w.size(1)), cfg.recipe, fmt)
        sw = _dynamic_scale(w, cfg.recipe, fmt)
        x8, _ = quantize(x, sx.reciprocal(), fmt)
        w8 = w if _is_fp8(w.dtype) else quantize(w, sw.reciprocal(), fmt)[0]
        out = mm_fp8(x8.reshape(-1, x8.size(-1)), w8, sx * sw, trans_b=True).reshape(
            *x.shape[:-1], w.size(0)
        )
        return (out + bias if bias.numel() else out), sx, sw

    meta = state.get_weight_meta(w)
    if not meta.w.initialized:
        meta.w.seed(w, fmt)
    if not meta.x.initialized:
        meta.x.seed(x, fmt)
    sx, sw = meta.x.scale.clone(), meta.w.scale.clone()
    x8, amax_x = quantize(x, sx.reciprocal(), fmt)
    if _is_fp8(w.dtype):
        w8, amax_w = w, None
    else:
        w8, amax_w = quantize(w, sw.reciprocal(), fmt)
    out = mm_fp8(x8.reshape(-1, x8.size(-1)), w8, sx * sw, trans_b=True).reshape(
        *x.shape[:-1], w.size(0)
    )
    if bias.numel():
        out = out + bias
    meta.x.update(amax_x, fmt)
    if amax_w is not None:
        meta.w.update(amax_w, fmt)
    meta.x.advance()
    if amax_w is not None:
        meta.w.advance()
    return out, sx, sw


class _LinearFp8(torch.autograd.Function):
    """The fp8 linear forward/backward pair (standard Function style).

    The forward runs inside ``fp8_autocast`` and captures the active
    fmt/recipe/meta on ``ctx``; the backward reads only that captured state, so
    ``loss.backward()`` may run after the context exits. The gradient is
    quantized once (E5M2 in hybrid) and both dX/dW GEMMs share it; the output
    masks come from ``needs_input_grad``.
    """

    @staticmethod
    def forward(ctx, x, w, bias):
        cfg = _current_config()
        out, sx, sw = fp8_linear_forward(x, w, bias, cfg)
        ctx.save_for_backward(x, w, sx, sw)
        ctx.fmt_bwd = cfg.fp8_format.bwd()
        ctx.recipe = cfg.recipe
        ctx.is_dynamic = isinstance(cfg.recipe, DynamicScaling)
        ctx.meta = None if ctx.is_dynamic else _state.get_weight_meta(w)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        x, w, _sx_fwd, _sw_fwd = ctx.saved_tensors
        fmt = ctx.fmt_bwd
        if ctx.is_dynamic:
            sg = _dynamic_scale(g, ctx.recipe, fmt)
            sw = _dynamic_scale(w, ctx.recipe, fmt)
            sx = _dynamic_scale(x, ctx.recipe, fmt)
        else:
            meta = ctx.meta
            if not meta.g.initialized:
                meta.g.seed(g, fmt)
            sg = meta.g.scale.clone()
            sw, sx = _sw_fwd, _sx_fwd
        g8, amax_g = quantize(g, sg.reciprocal(), fmt)
        x8, _ = quantize(x, sx.reciprocal(), fmt)
        w8 = w if _is_fp8(w.dtype) else quantize(w, sw.reciprocal(), fmt)[0]
        grad_x = mm_fp8(g8, w8, sg * sw)  # g8[m,n] @ w8[n,k] natural
        grad_w = mm_fp8(g8, x8, sg * sx, trans_a=True)  # g8.T @ x8
        grad_b = g.sum(0).to(torch.bfloat16)
        if not ctx.is_dynamic:
            meta.g.update(amax_g, fmt)
            meta.g.advance()
        return grad_x, grad_w, grad_b if ctx.needs_input_grad[2] else None


# ---------------------------------------------------------------------------
# aten::linear integration
# ---------------------------------------------------------------------------


def fp8_linear_enable(enabled: bool = True) -> None:
    """Toggle fp8 dispatch for aten::linear globally (the out-of-region default;
    ``fp8_autocast`` regions override it thread-locally)."""
    fp8_state().default_enabled = enabled


def fp8_linear_enabled() -> bool:
    """Whether fp8 dispatch is active right now (region config or global)."""
    return _active() is not None


def _fp8_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Shape guard for the fp8 path. Unlike a strict 16-alignment requirement,
    the kernels handle unaligned M/N via boundary checks (slower but correct) —
    so no whole-call bf16 fallback for small decode batches. Only the K-dimension
    contraction must match and the weight must be 2D."""
    return x.dim() >= 2 and w.dim() == 2 and x.size(-1) == w.size(1)


def _linear_cuda_impl(x: torch.Tensor, w: torch.Tensor, bias=None):
    if (
        _active() is not None
        and x.dtype is torch.bfloat16
        and w.dtype is torch.bfloat16
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
# aten::linear_backward after the fp8_autocast region exits). The fp8 backward
# is owned by _LinearFp8 with state captured at forward time, so loss.backward()
# works wherever it is called; the CUDA registration still covers inference_mode.
_lib_autograd = Library("aten", "IMPL", "AutogradCUDA")
_lib_autograd.impl("linear", _linear_cuda_impl)
