"""Kernel-level mask dimension support (2D, 3D, 4D)."""

import torch

from astrai.extension.ops.attention import attn_prefill
from tests.extension.conftest import D, skip_no_kernel


@skip_no_kernel
def test_kernel_accepts_2d_mask():
    """Kernel should accept 2D mask [batch, kv_len]."""
    batch, q_len, n_heads, n_kv_heads = 1, 8, 4, 1
    kv_len = 8
    q = torch.randn(batch, q_len, n_heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(batch, kv_len, dtype=torch.bool, device="cuda")
    mask[:, 4:] = False

    out = attn_prefill(q, k, v, mask=mask, is_causal=False)
    assert out.shape == (batch, q_len, n_heads, D)


@skip_no_kernel
def test_kernel_accepts_3d_mask():
    """Kernel should accept 3D mask [batch, q_len, kv_len]."""
    batch, q_len, n_heads, n_kv_heads = 1, 8, 4, 1
    kv_len = 8
    q = torch.randn(batch, q_len, n_heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(batch, q_len, kv_len, dtype=torch.bool, device="cuda")

    out = attn_prefill(q, k, v, mask=mask, is_causal=False)
    assert out.shape == (batch, q_len, n_heads, D)


@skip_no_kernel
def test_kernel_accepts_4d_mask():
    """Kernel should accept 4D mask [batch, n_heads, q_len, kv_len]."""
    batch, q_len, n_heads, n_kv_heads = 1, 8, 4, 1
    kv_len = 8
    q = torch.randn(batch, q_len, n_heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(batch, 1, q_len, kv_len, dtype=torch.bool, device="cuda")
    mask[:, :, :, 4:] = False

    out = attn_prefill(q, k, v, mask=mask, is_causal=False)
    assert out.shape == (batch, q_len, n_heads, D)


@skip_no_kernel
def test_4d_mask_matches_no_mask_when_all_true():
    """A 4D all-True mask should produce the same output as no mask."""
    batch, q_len, n_heads, n_kv_heads = 1, 8, 4, 1
    kv_len = 8
    q = torch.randn(batch, q_len, n_heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)

    out_no_mask = attn_prefill(q, k, v, mask=None, is_causal=False)
    mask = torch.ones(batch, 1, q_len, kv_len, dtype=torch.bool, device="cuda")
    out_with_mask = attn_prefill(q, k, v, mask=mask, is_causal=False)

    diff = (out_no_mask.float() - out_with_mask.float()).abs().max().item()
    assert diff == 0.0, f"4D all-True mask diff: {diff}"
