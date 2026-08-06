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
to a process-wide ``TorchNativeBackend`` singleton.

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). The backend returns ``[batch, seq_len, n_heads * head_dim]``.
"""

import contextvars
import enum
import importlib
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.attention_ops import (
    attn_paged_decode,
    attn_paged_prefill,
)
from astrai.factory import BaseFactory

if TYPE_CHECKING:
    from astrai.inference.core.cache import KVCache

_current_backend: contextvars.ContextVar["AttentionBackend"] = contextvars.ContextVar(
    "attn_backend"
)

_lock = threading.Lock()
_flash_available: Optional[bool] = None


def flash_attn_available() -> bool:
    """Return ``True`` if the optional ``flash-attn`` package is usable.

    ``flash-attn`` is not a hard dependency (declared only as an optional
    extra and imported lazily), so this is checked at first use and cached.
    The check is stronger than "import works": it also gates on the GPU
    compute capability for the installed major version and smoke-tests a
    real tiny kernel call, because wheels that import fine can still fail
    at the first actual invocation (wrong arch build, torch mismatch, or a
    missing ``flash_attn_func`` entry point).  It never raises.
    """
    global _flash_available
    if _flash_available is None:
        with _lock:
            if _flash_available is None:
                _flash_available = _flash_attn_check()
    return _flash_available


_flash_attn_module = None
_flash_attn_import_tried = False


def _get_flash_attn():
    """Lazily import and cache the optional ``flash_attn`` module.

    Uses ``importlib.import_module`` so no static import binds the name when
    the package is absent.  Returns the module object, or ``None`` if the
    package is not installed or cannot be imported.  Never raises.
    """
    global _flash_attn_module, _flash_attn_import_tried
    if not _flash_attn_import_tried:
        _flash_attn_import_tried = True
        try:
            _flash_attn_module = importlib.import_module("flash_attn")
        except Exception:
            _flash_attn_module = None
    return _flash_attn_module


def _flash_attn_check() -> bool:
    if not torch.cuda.is_available():
        return False
    fa = _get_flash_attn()
    if fa is None:
        return False

    # version + compute-capability gate:
    #   FlashAttention-2 kernels need sm_70+; FlashAttention-3 (tcgen05,
    #   sm_90/sm_100) needs sm_90+.
    try:
        major = int(fa.__version__.split(".")[0])
        cc = torch.cuda.get_device_capability()
        cc_num = cc[0] * 10 + cc[1]
    except Exception:
        major, cc_num = 0, 0
    if (major >= 3 and cc_num < 90) or (major < 3 and 0 < cc_num < 70):
        return False

    # smoke-test the real kernel: a wheel that imports but was built for a
    # different arch/torch fails here instead of at the first real forward.
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


def get_backend() -> "AttentionBackend":
    """Return the active backend for the current thread/context.

    Falls back to a ``TorchNativeBackend`` singleton when no backend
    has been activated via ``with``.
    """
    try:
        return _current_backend.get()
    except LookupError:
        return _default_backend


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
    if isinstance(backend, ATTN_BACKEND):
        instance = AttentionBackendFactory.create(backend.value)
    elif isinstance(backend, str):
        instance = AttentionBackendFactory.create(backend)
    elif isinstance(backend, type) and issubclass(backend, AttentionBackend):
        instance = backend()
    elif isinstance(backend, AttentionBackend):
        instance = backend
    else:
        raise TypeError(
            f"expected a registered name, ATTN_BACKEND, AttentionBackend type, "
            f"or instance, "
            f"got {type(backend).__name__}"
        )
    token = _current_backend.set(instance)
    try:
        yield instance
    finally:
        _current_backend.reset(token)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads to match Q heads for GQA."""
    bs, slen, n_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_heads, n_rep, head_dim)
        .reshape(bs, slen, n_heads * n_rep, head_dim)
    )


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Optional["KVCache"] = None,
    layer_id: int = 0,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
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
    backend = get_backend()
    return backend.forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal)


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
        if kv_cache is not None and q.size(1) == 1:
            return self.fwd_decode(q, k, v, kv_cache, layer_id, attn_mask, is_causal)
        return self.fwd_prefill(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

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
        if kv_cache is not None:
            kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
            kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v

            max_len = kv_cache.max_len
            indices = kv_cache.req_to_token[kv_cache.req_pool_indices, :max_len]
            # Zero out padding positions so gather never touches invalid slots.
            # Decode: attn_mask[:,0,0] is exactly the per-position validity
            # mask ([B, max_len], True=keep).  Prefill: fall back to seq_lens.
            if q.size(1) == 1 and attn_mask is not None and attn_mask.dim() == 4:
                pos_mask = attn_mask[:, 0, 0]
            else:
                pos_mask = (
                    torch.arange(max_len, device=q.device)[None, :]
                    < kv_cache.seq_lens[:, None]
                )
            indices = torch.where(pos_mask, indices, torch.zeros_like(indices))
            k = kv_cache.k_buffer[layer_id, indices]
            v = kv_cache.v_buffer[layer_id, indices]

        n_rep = q.size(2) // k.size(2)
        if n_rep > 1:
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        out = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            attn_mask,
            is_causal=is_causal,
        )
        out = out.permute(0, 2, 1, 3).contiguous().flatten(2)
        return out


_default_backend = TorchNativeBackend()


@AttentionBackendFactory.register(ATTN_BACKEND.CUDA.value)
class CudaBackend(AttentionBackend):
    """CUDA kernel backend with direct KV cache access.

    Decode path: writes K/V to the flat pool, then calls
    ``attn_paged_decode`` with req_to_token + kv_indptr.

    Prefill path: writes K/V to the flat pool, then calls
    ``attn_paged_prefill`` with ragged-batch support via qo_indptr +
    kv_indptr.

    ``kv_cache is None`` (training) is not handled — use
    ``TorchNativeBackend`` for training.

    Raises ``RuntimeError`` if the required kernel is not available.
    """

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

        kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
        kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v

        b = q.size(0)
        q_3d = q.squeeze(1)

        kv_indptr = kv_cache.kv_indptr

        out = attn_paged_decode(
            q_3d,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_indptr,
            kv_cache.max_len,
            is_causal=True,
            o_part_buf=kv_cache.decode_o_part,
            ml_part_buf=kv_cache.decode_ml_part,
        )
        return out.unsqueeze(1).flatten(2)

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

        kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
        kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v

        b = q.size(0)
        q_len = q.size(1)

        kv_indptr = kv_cache.kv_indptr
        qo_indptr = kv_cache.qo_indptr

        q_flat = q.reshape(b * q_len, q.size(2), q.size(3))

        out = attn_paged_prefill(
            q_flat,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_indptr,
            qo_indptr,
            attn_mask,
            q_len,
            is_causal=is_causal,
        )
        return out.reshape(b, q_len, q.size(2), q.size(3)).flatten(2)


@AttentionBackendFactory.register(ATTN_BACKEND.FLASH.value)
class FlashAttnBackend(AttentionBackend):
    """FlashAttention (FA2/FA3) backend via the optional ``flash-attn`` package.

    Uses the general ``flash_attn_func`` entry point for both prefill and
    single-token decode, mirroring ``TorchNativeBackend``'s KV-cache gather.
    This backend only does flash attention — inputs ``flash-attn`` cannot
    express (missing package, custom attention mask, fp32, unsupported
    head_dim) raise a clear error instead of silently falling back to torch.

    For a torch fallback, select ``TorchNativeBackend`` instead.
    """

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
        if kv_cache is not None:
            kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
            kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v

            max_len = kv_cache.max_len
            indices = kv_cache.req_to_token[kv_cache.req_pool_indices, :max_len]
            if q.size(1) == 1 and attn_mask is not None and attn_mask.dim() == 4:
                pos_mask = attn_mask[:, 0, 0]
            else:
                pos_mask = (
                    torch.arange(max_len, device=q.device)[None, :]
                    < kv_cache.seq_lens[:, None]
                )
            indices = torch.where(pos_mask, indices, torch.zeros_like(indices))
            k = kv_cache.k_buffer[layer_id, indices]
            v = kv_cache.v_buffer[layer_id, indices]

        n_rep = q.size(2) // k.size(2)
        if n_rep > 1:
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        if attn_mask is not None and not is_causal:
            raise ValueError(
                "FlashAttnBackend does not support a custom attention mask; "
                "use a causal mask or select TorchNativeBackend."
            )
        fa = _get_flash_attn()
        if fa is None:
            raise RuntimeError(
                "FlashAttnBackend requires the optional 'flash-attn' package. "
                "Install with `pip install flash-attn`."
            )
        out = fa.flash_attn_func(
            q.contiguous(), k.contiguous(), v.contiguous(), causal=is_causal
        )
        return out.contiguous().flatten(2)
