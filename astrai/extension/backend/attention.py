"""Attention backend abstraction with context-manager switching.

The backend encapsulates KV cache I/O and attention computation. The
attention module (GQA/MLA) keeps projections, rotary, QK-norm, gating,
and output projection; the backend handles everything from "write K/V
to cache" through "SDPA output".

Usage — mirroring ``torch.nn.attention.sdpa_kernel``:

    from astrai.extension import attn_backend, ATTN_BACKEND

    with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
        engine.generate("hello")

    # or with an instance:
    with attn_backend(TorchNativeBackend()):
        ...

    # or the shorthand (instance is itself a context manager):
    with TorchNativeBackend():
        ...

Thread-safe via ``contextvars`` — each scheduler thread gets its own
active backend. ``get_backend()`` returns the active one, falling back
to a process-wide default (cuda > flash > torch, overridable via
``ASTR_BACKEND``).

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). The backend returns ``[batch, seq_len, n_heads * head_dim]``.
"""

import contextvars
import enum
import functools
import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.loader import is_available
from astrai.extension.ops.attention import (
    attn_paged_decode,
    attn_paged_prefill,
)
from astrai.factory import BaseFactory

try:
    import flash_attn as _flash_attn
except Exception:
    _flash_attn = None

if TYPE_CHECKING:
    from astrai.inference.cache import KVCache


_default_backend: Optional["AttentionBackend"] = None
_default_backend_lock = threading.Lock()
_env_backend_name: Optional[str] = None
_env_backend: Optional["AttentionBackend"] = None
_current_backend: contextvars.ContextVar[Optional["AttentionBackend"]] = (
    contextvars.ContextVar("attn_backend", default=None)
)


@functools.lru_cache(maxsize=1)
def flash_attn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    fa = _flash_attn
    if fa is None:
        return False

    try:
        major = int(fa.__version__.split(".")[0])
        cc = torch.cuda.get_device_capability()
        cc_num = cc[0] * 10 + cc[1]
    except Exception:
        major, cc_num = 0, 0
    if (major >= 3 and cc_num < 90) or (major < 3 and 0 < cc_num < 70):
        return False

    try:
        if not hasattr(fa, "flash_attn_func"):
            return False
        x = torch.zeros(1, 1, 1, 64, device="cuda", dtype=torch.bfloat16)
        out = fa.flash_attn_func(x, x, x, causal=True)
        return bool(torch.isfinite(out).all().item())
    except Exception:
        return False


class ATTN_BACKEND(enum.Enum):
    """Backend selector enum, mirroring ``torch.nn.attention.SDPBackend``."""

    TORCH_NATIVE = "torch_native"
    CUDA = "cuda"
    FLASH = "flash"


def _priority_backends() -> list["AttentionBackend"]:
    """Available backends in priority order: cuda -> flash -> torch."""
    backends: list[AttentionBackend] = []
    if is_available("attn_paged_decode") and is_available("attn_paged_prefill"):
        backends.append(CudaBackend())
    if flash_attn_available():
        backends.append(FlashAttnBackend())
    backends.append(TorchNativeBackend())
    return backends


def _backend_supports(
    backend: "AttentionBackend",
    q: Tensor,
    kv_cache: Optional["KVCache"],
    attn_mask: Optional[Tensor],
    is_causal: bool,
    fwd: Optional[str],
) -> bool:
    """Whether ``backend`` can run this attention call.

    The CUDA kernels are bf16-only, support head_dim in 32/64/128/256, and
    need a KV cache (decode/prefill); everything else falls back to torch.
    """
    if isinstance(backend, CudaBackend):
        return (
            fwd in ("prefill", "decode")
            and kv_cache is not None
            and q.ndim == 3
            and q.dtype == torch.bfloat16
            and q.size(-1) in (32, 64, 128, 256)
            and is_available(f"attn_paged_{fwd}")
        )
    if isinstance(backend, FlashAttnBackend):
        if not flash_attn_available():
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        if fwd is not None:
            return q.ndim == 3 and hasattr(_flash_attn, "flash_attn_varlen_func")
        if attn_mask is None or is_causal:
            return True
        return attn_mask.dim() == 4
    return True


def _resolve_default_backend() -> "AttentionBackend":
    """Pick the highest-priority available backend (cuda -> flash -> torch).

    Resolved lazily on first ``get_backend()`` and cached.  Per-call
    capability fallback happens in ``attention()``, so the default is
    safe for training and fp32 models.
    """
    return _priority_backends()[0]


def _environment_backend() -> Optional["AttentionBackend"]:
    """Resolve the process-wide ``ASTR_BACKEND`` override, if configured."""
    global _env_backend, _env_backend_name
    name = os.environ.get("ASTR_BACKEND", "").strip().lower()
    if not name:
        return None
    if name != _env_backend_name:
        with _default_backend_lock:
            if name != _env_backend_name:
                try:
                    _env_backend = AttentionBackendFactory.create(name)
                except (ValueError, RuntimeError):
                    _env_backend = None
                _env_backend_name = name
    return _env_backend


def _resolve_backend(
    backend: Optional[Union[str, ATTN_BACKEND, "AttentionBackend", type]] = None,
) -> "AttentionBackend":
    """Resolve a backend configuration, defaulting to the process policy."""
    if backend is not None:
        if isinstance(backend, ATTN_BACKEND):
            return AttentionBackendFactory.create(backend.value)
        if isinstance(backend, str):
            return AttentionBackendFactory.create(backend)
        if isinstance(backend, type) and issubclass(backend, AttentionBackend):
            return backend()
        if isinstance(backend, AttentionBackend):
            return backend
        raise TypeError(
            f"expected a registered name, ATTN_BACKEND, AttentionBackend type, "
            f"or instance, got {type(backend).__name__}"
        )

    global _default_backend
    if _default_backend is None:
        with _default_backend_lock:
            if _default_backend is None:
                _default_backend = _resolve_default_backend()
    return _default_backend


def get_backend(
    use_default: bool = True,
) -> Optional["AttentionBackend"]:
    """Return the context override, optionally falling back to the process default.

    ``ASTR_BACKEND`` is a process-wide override and takes precedence over the
    context value. Pass ``use_default=False`` at request submission to retain
    only an environment override or the caller's :func:`attn_backend` value.
    """
    return (
        _environment_backend()
        or _current_backend.get()
        or (_resolve_backend() if use_default else None)
    )


@contextmanager
def attn_backend(backend: Union[str, ATTN_BACKEND, "AttentionBackend", type]):
    """Context manager to select an attention backend.

    Mirrors ``torch.nn.attention.sdpa_kernel``. Accepts an
    registered name, ``ATTN_BACKEND`` enum value, backend class, or instance.

    Examples::

        with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
            ...
        with attn_backend(TorchNativeBackend):
            ...
        with attn_backend(TorchNativeBackend()):
            ...
    """
    instance = _resolve_backend(backend)
    token = _current_backend.set(instance)
    try:
        yield instance
    finally:
        _current_backend.reset(token)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads to match Q heads for GQA."""
    if n_rep == 1:
        return x
    n_heads, head_dim = x.shape[-2:]
    return (
        x.unsqueeze(-2)
        .expand(*x.shape[:-2], n_heads, n_rep, head_dim)
        .reshape(*x.shape[:-2], n_heads * n_rep, head_dim)
    )


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Optional["KVCache"] = None,
    layer_id: int = 0,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    fwd: Optional[str] = None,
) -> Tensor:
    """Functional attention entry point — mirrors ``F.scaled_dot_product_attention``.

    Delegates to the active backend (set via ``with attn_backend(...)``).
    Handles KV cache I/O, GQA head expansion, and causal masking so the
    caller only needs to provide projected q/k/v.

    Args:
        q: [batch, q_len, n_heads, head_dim] (blhd)
        k: [batch, q_len, n_kv_heads, head_dim] (blhd)
        v: [batch, q_len, n_kv_heads, head_dim] (blhd)
        kv_cache: cache dataclass, or None for training (no cache).
        layer_id: transformer layer index for buffer access.
        attn_mask: pre-built attention mask (SDPA-compatible).
        is_causal: whether to apply causal masking.

    Returns:
        [batch, q_len, n_heads * head_dim]
    """
    explicit = get_backend(use_default=False)
    backend = get_backend()
    if fwd is None and explicit is None:
        backend = TorchNativeBackend()
    if not _backend_supports(backend, q, kv_cache, attn_mask, is_causal, fwd):
        if explicit is not None:
            raise RuntimeError(
                f"Explicitly-set backend {type(backend).__name__} cannot "
                f"handle this attention call (shape={q.shape}, "
                f"dtype={q.dtype}, kv_cache={'none' if kv_cache is None else 'present'}, "
                f"attn_mask={'none' if attn_mask is None else 'present'}). "
                f"Remove the attn_backend() context or switch to a compatible backend."
            )
        for candidate in _priority_backends():
            if isinstance(candidate, type(backend)):
                continue
            if _backend_supports(candidate, q, kv_cache, attn_mask, is_causal, fwd):
                backend = candidate
                break
    return backend.forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal, fwd)


class AttentionBackend(ABC):
    """Abstract base for attention computation strategies.

    Subclasses implement ``fwd_decode`` (q_len == 1, with cache) and
    ``fwd_prefill`` (q_len > 1, with or without cache). The public
    ``forward`` method dispatches based on q_len.

    Three equivalent ways to activate a backend::

        with attn_backend(ATTN_BACKEND.TORCH_NATIVE):  # enum
            ...
        with attn_backend(TorchNativeBackend):          # class
            ...
        with TorchNativeBackend():                       # instance
            ...
    """

    def __enter__(self) -> "AttentionBackend":
        self._token = _current_backend.set(self)
        return self

    def __exit__(self, *exc) -> None:
        _current_backend.reset(self._token)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        fwd: Optional[str] = None,
    ) -> Tensor:
        """Dispatch to decode or extend based on q_len.

        Args:
            q: [batch, q_len, n_heads, head_dim]
            k: [batch, q_len, n_kv_heads, head_dim]
            v: [batch, q_len, n_kv_heads, head_dim]
            kv_cache: cache dataclass, or None for training (no cache).
            layer_id: transformer layer index for buffer access.
            attn_mask: pre-built attention mask compatible with SDPA.
            is_causal: whether to apply causal masking.

        Returns:
            [batch, q_len, n_heads * head_dim]
        """
        if fwd == "decode":
            return self.fwd_decode(q, k, v, kv_cache, layer_id, attn_mask, is_causal)
        if fwd == "prefill" or fwd is None:
            return self.fwd_prefill(q, k, v, kv_cache, layer_id, attn_mask, is_causal)
        raise ValueError(f"unsupported attention forward mode: {fwd}")

    @abstractmethod
    def fwd_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Single-token decode with KV cache."""

    @abstractmethod
    def fwd_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Multi-token prefill or training forward."""

    @staticmethod
    def supports_graph() -> bool:
        """Return True if this backend supports CUDA-graph capture.

        Override in subclasses that can run under ``torch.cuda.graph``.

        Called on the *active* backend instance (or its class) — a cheap
        boolean check with no side-effects.
        """
        return False


class AttentionBackendFactory(BaseFactory[AttentionBackend]):
    """Factory for registered attention backends."""


@AttentionBackendFactory.register(ATTN_BACKEND.TORCH_NATIVE.value)
class TorchNativeBackend(AttentionBackend):
    """Reference backend using torch SDPA with indirect KV cache indexing.

    Writes new K/V into the cache buffers, gathers the full sequence K/V
    via ``req_to_token`` indirect indexing, then calls
    ``F.scaled_dot_product_attention``.

    For training (``kv_cache is None``), skips cache I/O entirely and
    runs SDPA directly on the projected q/k/v.
    """

    @staticmethod
    def supports(**kwargs) -> bool:
        return True

    def fwd_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        return self._forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    def fwd_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        return self._forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    def _forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if q.ndim == 4:
            n_rep = q.size(2) // k.size(2)
            if n_rep > 1:
                k = repeat_kv(k, n_rep)
                v = repeat_kv(v, n_rep)
            return (
                F.scaled_dot_product_attention(
                    q.permute(0, 2, 1, 3),
                    k.permute(0, 2, 1, 3),
                    v.permute(0, 2, 1, 3),
                    attn_mask,
                    is_causal=is_causal,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        if kv_cache is None or kv_cache.qo_indptr is None:
            raise ValueError("packed attention requires KV cache metadata")
        kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
        kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v
        outputs = []
        n_rep = q.size(1) // k.size(1)
        for i in range(kv_cache.req_pool_indices.numel()):
            q_start = int(kv_cache.qo_indptr[i])
            q_end = int(kv_cache.qo_indptr[i + 1])
            indices = kv_cache.req_to_token[
                kv_cache.req_pool_indices[i], : kv_cache.seq_lens[i]
            ]
            k_i = kv_cache.k_buffer[layer_id, indices]
            v_i = kv_cache.v_buffer[layer_id, indices]
            if n_rep > 1:
                k_i = repeat_kv(k_i, n_rep)
                v_i = repeat_kv(v_i, n_rep)
            q_len = q_end - q_start
            kv_len = k_i.size(0)
            q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)
            causal_mask = q_pos[:, None] >= torch.arange(kv_len, device=q.device)
            out = F.scaled_dot_product_attention(
                q[q_start:q_end].transpose(0, 1).unsqueeze(0),
                k_i.transpose(0, 1).unsqueeze(0),
                v_i.transpose(0, 1).unsqueeze(0),
                attn_mask=causal_mask,
            )
            outputs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outputs)


@AttentionBackendFactory.register(ATTN_BACKEND.CUDA.value)
class CudaBackend(AttentionBackend):
    """CUDA kernel backend with direct KV cache access.

    Decode path: writes K/V to the flat pool, then calls
    ``attn_paged_decode`` with req_to_token + kv_indptr.

    Prefill path: writes K/V to the flat pool, then calls
    ``attn_paged_prefill`` with ragged-batch support via qo_indptr +
    kv_indptr.

    ``kv_cache is None`` (training) raises — the per-call fallback to
    torch SDPA for training / fp32 / unsupported head_dim happens in the
    ``attention()`` entry point.

    Raises ``RuntimeError`` if the required kernel is not available.
    """

    @staticmethod
    def supports(**kwargs) -> bool:
        head_dim = kwargs.get("head_dim", -1)
        return (
            torch.cuda.is_available()
            and head_dim in (32, 64, 128, 256)
            and is_available("attn_paged_decode")
            and is_available("attn_paged_prefill")
        )

    @staticmethod
    def supports_graph() -> bool:
        return True

    def fwd_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is None:
            raise RuntimeError("CudaBackend does not support training (kv_cache=None)")

        kv_indptr = kv_cache.kv_indptr

        out = attn_paged_decode(
            q,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_indptr,
            new_k=k,
            new_v=v,
            is_causal=True,
            o_part_buf=kv_cache.decode_o_part,
            ml_part_buf=kv_cache.decode_ml_part,
            out_buf=kv_cache.decode_out,
        )
        return out

    def fwd_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is None:
            raise RuntimeError("CudaBackend does not support training (kv_cache=None)")

        loc = kv_cache.out_cache_loc
        kv_cache.k_buffer[layer_id, loc] = k
        kv_cache.v_buffer[layer_id, loc] = v

        out = attn_paged_prefill(
            q,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_cache.kv_indptr,
            kv_cache.qo_indptr,
            kv_cache.q_tile_to_batch,
            kv_cache.q_tile_to_index,
            attn_mask,
            is_causal=is_causal,
        )
        return out


@AttentionBackendFactory.register(ATTN_BACKEND.FLASH.value)
class FlashAttnBackend(AttentionBackend):
    """FlashAttention backend via the optional ``flash-attn`` package.

    Decode (q_len=1, contiguous cache): uses ``flash_attn_with_kvcache``,
    which reads K/V directly from the flat pool via cache_batch_idx +
    cache_seqlens — no materialized KV gather.

    Prefill / non-contiguous decode: falls back to KV gather +
    ``flash_attn_func``.
    """

    @staticmethod
    def supports(**kwargs) -> bool:
        return flash_attn_available()

    def fwd_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        return self._forward_packed(q, k, v, kv_cache, layer_id)

    def fwd_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if q.ndim == 3:
            return self._forward_packed(q, k, v, kv_cache, layer_id)
        return self._forward_dense(q, k, v, attn_mask, is_causal)

    def _forward_dense(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        n_rep = q.size(2) // k.size(2)
        if n_rep > 1:
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        if attn_mask is not None and not is_causal and attn_mask.dim() != 4:
            raise ValueError(
                "FlashAttnBackend does not support a custom attention mask; "
                "use a causal mask or select TorchNativeBackend."
            )
        fa = _flash_attn
        if fa is None:
            raise RuntimeError(
                "FlashAttnBackend requires the optional 'flash-attn' package. "
                "Install with `pip install flash-attn`."
            )
        out = fa.flash_attn_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            causal=is_causal or (attn_mask is not None and attn_mask.dim() == 4),
        )
        return out.contiguous()

    def _forward_packed(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: "KVCache",
        layer_id: int,
    ) -> Tensor:
        fa = _flash_attn
        if fa is None or not hasattr(fa, "flash_attn_varlen_func"):
            raise RuntimeError("packed inference requires flash_attn_varlen_func")
        kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
        kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v
        page_table = kv_cache.req_to_token[
            kv_cache.req_pool_indices, : kv_cache.max_len
        ]
        positions = torch.arange(kv_cache.max_len, device=q.device)
        indices = page_table[positions.unsqueeze(0) < kv_cache.seq_lens.unsqueeze(1)]
        k_flat = kv_cache.k_buffer[layer_id, indices].contiguous()
        v_flat = kv_cache.v_buffer[layer_id, indices].contiguous()
        out = fa.flash_attn_varlen_func(
            q.contiguous(),
            k_flat,
            v_flat,
            kv_cache.qo_indptr,
            kv_cache.kv_indptr,
            int((kv_cache.qo_indptr[1:] - kv_cache.qo_indptr[:-1]).max()),
            int(kv_cache.seq_lens.max()),
            dropout_p=0.0,
            causal=True,
        )
        return out
