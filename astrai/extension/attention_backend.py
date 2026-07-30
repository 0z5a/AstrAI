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
import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.inference.core.cache import KVCache

_current_backend: contextvars.ContextVar["AttentionBackend"] = contextvars.ContextVar(
    "attn_backend"
)


class ATTN_BACKEND(enum.Enum):
    """Backend selector enum, mirroring ``torch.nn.attention.SDPBackend``."""

    TORCH_NATIVE = "torch_native"


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
def attn_backend(backend: Union[ATTN_BACKEND, "AttentionBackend", type]):
    """Context manager to select an attention backend.

    Mirrors ``torch.nn.attention.sdpa_kernel``. Accepts an
    ``ATTN_BACKEND`` enum value, a backend class, or a backend instance.

    Examples::

        with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
            ...
        with attn_backend(TorchNativeBackend):
            ...
        with attn_backend(TorchNativeBackend()):
            ...
    """
    if isinstance(backend, ATTN_BACKEND):
        instance = _BACKEND_REGISTRY[backend]()
    elif isinstance(backend, type) and issubclass(backend, AttentionBackend):
        instance = backend()
    elif isinstance(backend, AttentionBackend):
        instance = backend
    else:
        raise TypeError(
            f"expected ATTN_BACKEND, AttentionBackend type, or instance, "
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


class AttentionBackend(ABC):
    """Abstract base for attention computation strategies.

    Subclasses implement ``forward_decode`` (q_len == 1, with cache) and
    ``forward_extend`` (q_len > 1, with or without cache). The public
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
        kv_cache: Optional[KVCache],
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
            return self.forward_decode(
                q, k, v, kv_cache, layer_id, attn_mask, is_causal
            )
        return self.forward_extend(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    @abstractmethod
    def forward_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Single-token decode with KV cache."""

    @abstractmethod
    def forward_extend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Multi-token prefill or training forward."""


class TorchNativeBackend(AttentionBackend):
    """Reference backend using torch SDPA with indirect KV cache indexing.

    Writes new K/V into the cache buffers, gathers the full sequence K/V
    via ``req_to_token`` indirect indexing, then calls
    ``F.scaled_dot_product_attention``.

    For training (``kv_cache is None``), skips cache I/O entirely and
    runs SDPA directly on the projected q/k/v.
    """

    def forward_decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        return self._forward(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    def forward_extend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional[KVCache],
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
        kv_cache: Optional[KVCache],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if kv_cache is not None:
            kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
            kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v

            max_len = kv_cache.seq_lens.max()
            indices = kv_cache.req_to_token[kv_cache.req_pool_indices, :max_len]
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

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask, is_causal=is_causal)
        out = out.permute(0, 2, 1, 3).contiguous().flatten(2)
        return out


_default_backend = TorchNativeBackend()

_BACKEND_REGISTRY: dict[ATTN_BACKEND, type[AttentionBackend]] = {
    ATTN_BACKEND.TORCH_NATIVE: TorchNativeBackend,
}
