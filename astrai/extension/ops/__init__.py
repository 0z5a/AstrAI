"""Stateless wrappers around compiled extension kernels."""

from astrai.extension.ops.attention import (
    TensorLayout,
    attn_decode,
    attn_paged_decode,
    attn_paged_prefill,
    attn_prefill,
)
from astrai.extension.ops.gemm import bf16_gemm
from astrai.extension.ops.rotary import rotary_emb
from astrai.extension.ops.swiglu import bf16_swiglu

__all__ = [
    "TensorLayout",
    "attn_decode",
    "attn_paged_decode",
    "attn_paged_prefill",
    "attn_prefill",
    "bf16_gemm",
    "bf16_swiglu",
    "rotary_emb",
]
