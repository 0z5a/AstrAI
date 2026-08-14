"""FP8 training state: per-tensor scales, amax history, delayed scaling.

TE-style (TransformerEngine) delayed scaling:
- weight tensors carry an ``FP8TensorMeta`` keyed by (data_ptr, shape) with a
  fixed scale derived from a 16-step amax history window;
- activations/gradients reuse the quantize kernel's free atomic amax, delayed
  one step (scale updated after each call, used by the next call);
- ``fp8_autocast()`` context manager toggles fp8 dispatch (like
  ``torch.autocast``) and advances the scale-update counter once per step.
  Entering it also ensures the aten::linear CUDA impl is registered, so
  ``import astrai.extension.fp8_dispatch`` is not required by callers.
"""

from contextlib import contextmanager

import torch

E4M3_MAX = 448.0

# FP8 GEMM layout: D = A_SCALE * B_SCALE * A * B, so the per-tensor scales are
# amax/448 (e4m3) and the quantization divides by scale (multiplies by 1/scale).


class FP8TensorMeta:
    """Scales + amax state for one weight tensor and its paired activations.

    - weight: delayed scale from a 16-step amax history window (TE style)
    - x/g: delayed one step, reuse the quantize kernel's free atomic amax
    """

    __slots__ = (
        "scale",
        "scale_inv",
        "amax_history",
        "idx",
        "x_scale",
        "x_scale_inv",
        "g_scale",
        "g_scale_inv",
    )

    def __init__(self, device: torch.device, update_interval: int):
        self.scale = torch.ones(1, device=device, dtype=torch.float32)
        self.scale_inv = torch.ones(1, device=device, dtype=torch.float32)
        self.amax_history = torch.ones(
            update_interval, device=device, dtype=torch.float32
        )
        self.idx = 0
        self.x_scale = torch.ones(1, device=device, dtype=torch.float32)
        self.x_scale_inv = torch.ones(1, device=device, dtype=torch.float32)
        self.g_scale = torch.ones(1, device=device, dtype=torch.float32)
        self.g_scale_inv = torch.ones(1, device=device, dtype=torch.float32)

    def record(self, amax: torch.Tensor) -> None:
        """Push the latest amax into the ring buffer (device-side copy, no sync)."""
        self.amax_history[self.idx] = amax.reshape(())
        self.idx = (self.idx + 1) % self.amax_history.numel()

    def refresh(self) -> None:
        """Recompute scale from the amax history window (delayed scaling)."""
        amax = self.amax_history.max()
        if amax > 0:
            self.scale.copy_(amax / E4M3_MAX)
            self.scale_inv.copy_(E4M3_MAX / amax)


class FP8State:
    """Global fp8 training state, TE-style."""

    def __init__(self, update_interval: int = 16):
        self.enabled = False
        self.update_interval = update_interval
        self.step_count = 0
        self._metas: dict[tuple, FP8TensorMeta] = {}
        self._last_device: torch.device | None = None

    def _get_device(self, t: torch.Tensor) -> torch.device:
        if self._last_device is None:
            self._last_device = t.device
        return t.device

    def get_weight_meta(self, w: torch.Tensor) -> FP8TensorMeta:
        key = (w.data_ptr(), w.shape, w.dtype)
        meta = self._metas.get(key)
        if meta is None:
            meta = FP8TensorMeta(self._get_device(w), self.update_interval)
            self._metas[key] = meta
        return meta

    def step(self) -> None:
        """Advance the counter and refresh all weight scales every N steps."""
        self.step_count += 1
        if self.step_count % self.update_interval == 0:
            for meta in self._metas.values():
                meta.refresh()

    def reset(self) -> None:
        self.enabled = False
        self.step_count = 0
        self._metas.clear()
        self._last_device = None


# Global singleton: autograd backward runs on the engine worker threads, so
# thread-local state would lose the fp8 flag during loss.backward(). The GIL
# protects Python-side mutation; the CUDA kernels take their own mutex.
_state = FP8State()


def fp8_state() -> FP8State:
    return _state


@contextmanager
def fp8_autocast(enabled: bool = True, update_interval: int = 16):
    """Autocast-style context: fp8 linear dispatch on this thread.

    Usage::

        with fp8_autocast(enabled=True):
            logits = model(input_ids)   # aten::linear -> fp8 path
            loss.backward()

    The scale-update counter advances once per ``enter`` (one training step),
    refreshing weight scales from their amax history every ``update_interval``.
    """
    state = fp8_state()
    prev_enabled = state.enabled
    prev_interval = state.update_interval
    state.enabled = enabled
    state.update_interval = update_interval
    try:
        if enabled:
            state.step()
        yield
    finally:
        state.enabled = prev_enabled
        state.update_interval = prev_interval


def _update_delayed_scale(scale, scale_inv, amax) -> None:
    """scale = amax / 448 for the *next* call (device-side, no sync)."""
    amax_f = amax.reshape(()).to(torch.float32).clamp_min(1e-12)
    scale.copy_(amax_f / E4M3_MAX)
    scale_inv.copy_(E4M3_MAX / amax_f)


def fp8_linear_forward(x: torch.Tensor, w: torch.Tensor, bias=None):
    """TE-style scaled fp8 linear forward (called from the aten::linear impl).

    x uses the delayed scale of its paired weight meta (amax from the previous
    forward of this linear); the quantize kernel emits the current amax for the
    next step. No extra abs/max reduce.
    """
    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)
    state = fp8_state()
    mod = _mod()
    meta = state.get_weight_meta(w)
    amax_x = torch.empty(1, device=x.device, dtype=torch.float32)
    amax_w = torch.empty(1, device=x.device, dtype=torch.float32)
    out = mod.fp8_linear_forward_scaled(
        x,
        w,
        bias,
        meta.x_scale,
        meta.scale,
        meta.x_scale_inv,
        meta.scale_inv,
        amax_x,
        amax_w,
    )
    meta.record(amax_w)
    _update_delayed_scale(meta.x_scale, meta.x_scale_inv, amax_x)
    return out


def fp8_linear_backward(g, x, w, masks):
    """TE-style scaled fp8 linear backward (called from aten::linear_backward)."""
    state = fp8_state()
    mod = _mod()
    meta = state.get_weight_meta(w)
    amax_g = torch.empty(1, device=g.device, dtype=torch.float32)
    out = mod.fp8_linear_backward_scaled(
        g,
        x,
        w,
        list(masks),
        meta.g_scale,
        meta.scale,
        meta.x_scale,
        meta.g_scale_inv,
        meta.scale_inv,
        meta.x_scale_inv,
        amax_g,
    )
    _update_delayed_scale(meta.g_scale, meta.g_scale_inv, amax_g)
    return out


def _mod():
    from astrai.extension.loader import get_module

    return get_module("fp8_mm")
