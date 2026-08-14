"""FP8 training: scaling state and aten::linear dispatch.

Layered (see also ``fp8_ops.py`` for the CUDA interface adapter):

1. Kernel interface: "fp8_ops" — the only module touching the pybind.
2. Training state (this module): per-tensor scales, amax history, delayed
   scaling, and the ``fp8_autocast`` context (TE-style, like
   ``torch.autocast``).
3. aten::linear integration (this module): registers the CUDA impl and the
   M/N alignment guard.

Usage::

    from astrai.extension.fp8 import fp8_autocast

    with fp8_autocast(enabled=True):
        logits = model(input_ids)
        loss.backward()

Importing this module registers the aten::linear CUDA implementation.
"""

from contextlib import contextmanager

import torch
from torch.library import Library

from astrai.extension.fp8_ops import (
    linear_backward_scaled,
    linear_forward_scaled,
)

E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Layer 2: training state (scales, amax history, delayed scaling, autocast)
# ---------------------------------------------------------------------------


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
    meta = state.get_weight_meta(w)
    amax_x = torch.empty(1, device=x.device, dtype=torch.float32)
    amax_w = torch.empty(1, device=x.device, dtype=torch.float32)
    out = linear_forward_scaled(
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
    meta = state.get_weight_meta(w)
    amax_g = torch.empty(1, device=g.device, dtype=torch.float32)
    out = linear_backward_scaled(
        g,
        x,
        w,
        masks,
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


# ---------------------------------------------------------------------------
# Layer 3: aten::linear integration
# ---------------------------------------------------------------------------


def fp8_linear_enable(enabled: bool = True) -> None:
    """Toggle fp8 dispatch for aten::linear (global; backward runs on engine
    worker threads, so a thread-local flag would be lost during backward)."""
    fp8_state().enabled = enabled


def fp8_linear_enabled() -> bool:
    return fp8_state().enabled


def _fp8_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    """cuBLASLt fp8 requires M % 16 == 0 and N % 16 == 0 (K is padded)."""
    m = x.numel() // x.size(-1)
    return m % 16 == 0 and w.size(0) % 16 == 0


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
    if (
        fp8_linear_enabled()
        and weight.dtype == torch.bfloat16
        and _fp8_supported(grad_output, weight)
    ):
        return fp8_linear_backward(grad_output, input_tensor, weight, list(output_mask))
    compute_dtype = weight.dtype
    grad = grad_output.to(compute_dtype)
    grad_2d = grad.reshape(-1, weight.size(0))
    input_2d = input_tensor.reshape(-1, input_tensor.size(-1)).to(compute_dtype)
    grad_input = (
        torch.mm(grad_2d, weight)
        if output_mask[0]
        else torch.empty(0, device=input_tensor.device, dtype=input_tensor.dtype)
    )
    grad_weight = (
        torch.mm(grad_2d.t(), input_2d)
        if output_mask[1]
        else torch.empty(0, device=input_tensor.device, dtype=input_tensor.dtype)
    )
    grad_bias = (
        grad.sum(dim=0)
        if output_mask[2]
        else torch.empty(0, device=input_tensor.device, dtype=input_tensor.dtype)
    )
    return grad_input.reshape_as(input_tensor), grad_weight, grad_bias


_lib = Library("aten", "IMPL", "CUDA")
_lib.impl("linear", _linear_cuda_impl)
_lib.impl("linear_backward", _linear_backward_cuda_impl)
