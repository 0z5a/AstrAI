"""CUDA kernel wrappers, operator dispatch, and backend selection.

Public API:
    - ``attention``, ``linear``, ``swiglu``, ``apply_rotary_emb`` — op
      families with safe torch fallbacks (see ``astrai.extension.backend``)
    - ``attn_decode`` / ``attn_prefill`` / ``attn_paged_decode`` /
      ``attn_paged_prefill`` — direct attention kernel wrappers
    - ``bf16_gemm`` / ``bf16_swiglu`` — directly callable linear/MLP kernels
    - ``AttentionBackend`` / ``TorchNativeBackend`` / ``CudaBackend`` /
      ``FlashAttnBackend`` — attention backend strategies
    - ``resolve`` / ``explain`` / ``op_backend`` / ``env_mode`` — the shared
      operator dispatcher (see ``astrai.extension.dispatch``)

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). Scale is always ``1/sqrt(head_dim)``. Wrapper functions call their
compiled CUDA kernels directly; fallback is the backend's responsibility.
"""

from astrai.extension.backend import (
    ATTN_BACKEND,
    AttentionBackend,
    AttentionBackendFactory,
    CudaBackend,
    FlashAttnBackend,
    TorchNativeBackend,
    apply_rotary_emb,
    attention,
    attn_backend,
    get_backend,
    linear,
    swiglu,
)
from astrai.extension.dispatch import (
    Axes,
    ExplicitSelectionError,
    ImplRecord,
    Resolution,
    Spec,
    axis,
    env_mode,
    explain,
    explain_plan,
    op_backend,
    register_env_alias,
    register_family,
    resolve,
    resolve_plan,
    tensor_axes,
)
from astrai.extension.loader import KERNEL_NAMES, is_available
from astrai.extension.ops import (
    TensorLayout,
    attn_decode,
    attn_paged_decode,
    attn_prefill,
    bf16_gemm,
    bf16_swiglu,
)

__all__ = [
    "ATTN_BACKEND",
    "AttentionBackend",
    "AttentionBackendFactory",
    "CudaBackend",
    "TorchNativeBackend",
    "FlashAttnBackend",
    "TensorLayout",
    "attention",
    "attn_backend",
    "get_backend",
    "linear",
    "swiglu",
    "attn_decode",
    "attn_paged_decode",
    "attn_prefill",
    "bf16_gemm",
    "bf16_swiglu",
    "is_available",
    "KERNEL_NAMES",
    "apply_rotary_emb",
    "Axes",
    "ExplicitSelectionError",
    "ImplRecord",
    "Resolution",
    "Spec",
    "axis",
    "env_mode",
    "explain",
    "explain_plan",
    "op_backend",
    "register_env_alias",
    "register_family",
    "resolve",
    "resolve_plan",
    "tensor_axes",
]
