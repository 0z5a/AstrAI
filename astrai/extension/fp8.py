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
    ``[hist[n] | scale | counter]`` (views). The quantize kernel's last-finishing
    block records the measured amax into ``hist[idx]``, reduces the window and
    publishes the next scale entirely on device — the Python-side write/max/write
    chain is gone. The counter slot stays int32-zero (float bits) between
    launches; ``idx`` advances host-side each step.
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
        """Rotate to the next history slot after an in-kernel finalize."""
        self.idx = (self.idx + 1) % self.hist.numel()

    def seed(self, t: torch.Tensor, fmt: str) -> None:
        amax = t.abs().amax().to(torch.float32).clamp_min(1e-12)
        self.hist.fill_(amax)
        self.scale.copy_(self.recipe.scale_from_history(self.hist, fmt))
        self.initialized = True


class FP8TensorMeta:
    """Per-weight delayed-scaling state: one ring per operand role (``w``/``x``/
    ``g``). Fused kernels record amax while quantizing, so the scale used at step
    N reflects amax from steps < N. DynamicScaling never allocates a meta — it
    measures the current amax inline.
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
        self.default_enabled = False
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

    Pure FP8 path for both recipes: quantize x/w with the active scales, run the
    pre-quantized GEMM. Delayed scaling finalizes the rings inside the quantize
    kernels (amax folded into the window, next scale published on device);
    dynamic scaling measures the current amax itself. Training quantizes the
    weight every step (the optimizer bumps its version, so there is no cast
    cache, matching ``cached_cast``-less behavior).
    """
    state = fp8_state()
    if cfg is None:
        cfg = _current_config()
    fmt = cfg.fp8_format.fwd()
    margin = cfg.recipe.margin
    if bias is None:
        bias = _empty_bias(x)
    if isinstance(cfg.recipe, DynamicScaling):  # measure-then-quantize, no state
        sx = _dynamic_scale(x.reshape(-1, w.size(1)), cfg.recipe, fmt)
        sw = _dynamic_scale(w, cfg.recipe, fmt)
        out, *_ = linear_forward_fp8(x, w, bias, sx, sw, fmt)
        return out

    meta = state.get_weight_meta(w)
    if not meta.w.initialized:
        meta.w.seed(w, fmt)
    if not meta.x.initialized:
        meta.x.seed(x, fmt)
    # The kernel finalizes each ring in-kernel (overwriting the scale slot), so
    # the w/x scales are taken from the ring before the quantize.
    if w.dtype is not torch.bfloat16:  # static pre-quantized weight
        w_arg, sw_arg, w_ring = w, meta.w.scale, None
    else:
        w_arg, sw_arg, w_ring = w, meta.w.scale, meta.w.state
    out, _x8, _w8, _ax, _aw = linear_forward_fp8(
        x,
        w_arg,
        bias,
        meta.x.scale,
        sw_arg,
        fmt,
        None,
        meta.x.state,
        meta.x.idx,
        margin,
        w_ring,
        meta.w.idx,
        margin,
    )
    meta.x.advance()
    if w_ring is not None:
        meta.w.advance()
    return out


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
        out = fp8_linear_forward(x, w, bias, cfg)
        ctx.save_for_backward(x, w)
        ctx.fmt_bwd = cfg.fp8_format.bwd()
        ctx.recipe = cfg.recipe
        ctx.is_dynamic = isinstance(cfg.recipe, DynamicScaling)
        ctx.meta = None if ctx.is_dynamic else _state.get_weight_meta(w)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        fmt = ctx.fmt_bwd
        # Per-recipe scale/ring selection; both branches share one call below.
        if ctx.is_dynamic:
            sg = _dynamic_scale(g, ctx.recipe, fmt)
            sw = _dynamic_scale(w, ctx.recipe, fmt)
            sx = _dynamic_scale(x, ctx.recipe, fmt)
            ring, idx = None, 0
        else:
            meta = ctx.meta
            if not meta.g.initialized:
                meta.g.seed(g, fmt)
            sg, ring, idx = meta.g.scale, meta.g.state, meta.g.idx
            sw, sx = meta.w.scale, meta.x.scale
        grad_x, grad_w, grad_b, _amax_g = linear_backward_fp8(
            g,
            x,
            w,
            list(ctx.needs_input_grad),
            sg,
            sw,
            sx,
            fmt,
            ring,
            idx,
            ctx.recipe.margin,
        )
        if not ctx.is_dynamic:
            meta.g.advance()  # the g quantize kernel finalized the ring in-kernel
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
