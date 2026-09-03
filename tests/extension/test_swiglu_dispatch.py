import logging

import pytest
import torch
import torch.nn.functional as F

from astrai.extension import is_available, swiglu
from astrai.model.components.mlp import MLP

SWIGLU_AVAILABLE = (
    torch.cuda.is_available()
    and is_available("bf16_swiglu")
    and torch.cuda.get_device_capability() >= (8, 0)
)
skip_no_swiglu = pytest.mark.skipif(
    not SWIGLU_AVAILABLE,
    reason="BF16 SwiGLU requires a built kernel and compute capability 8.0+",
)


def reference_swiglu(x, up_weight, gate_weight):
    return F.linear(x, up_weight) * F.silu(F.linear(x, gate_weight))


def test_cpu_and_training_calls_fall_back_with_gradients(monkeypatch):
    monkeypatch.setenv("ASTRAI_SWIGLU", "1")
    x = torch.randn(2, 8, requires_grad=True)
    up_weight = torch.randn(4, 8, requires_grad=True)
    gate_weight = torch.randn(4, 8, requires_grad=True)
    actual = swiglu(x, up_weight, gate_weight)
    expected = reference_swiglu(x, up_weight, gate_weight)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None
    assert up_weight.grad is not None
    assert gate_weight.grad is not None


def test_invalid_mode_warns_and_uses_auto(monkeypatch, caplog):
    monkeypatch.setenv("ASTRAI_SWIGLU", "invalid-test-mode")
    with caplog.at_level(logging.WARNING):
        actual = swiglu(torch.randn(2, 8), torch.randn(4, 8), torch.randn(4, 8))
    assert actual.shape == (2, 4)
    assert "using auto" in caplog.text


def test_mlp_routes_through_swiglu_backend(monkeypatch):
    sentinel = torch.randn(2, 4)

    def fake_swiglu(x, up_weight, gate_weight):
        assert x.shape == (2, 3)
        assert up_weight.shape == gate_weight.shape == (4, 3)
        return sentinel

    monkeypatch.setattr("astrai.model.components.mlp.swiglu", fake_swiglu)
    layer = MLP(3, 4)
    output = layer(torch.randn(2, 3))
    assert output["hidden_states"].shape == (2, 3)


@skip_no_swiglu
def test_mode_zero_disables_fused_kernel(monkeypatch):
    monkeypatch.setenv("ASTRAI_SWIGLU", "0")
    x = torch.randn(4, 1536, device="cuda", dtype=torch.bfloat16) * 0.1
    up_weight = torch.randn(6912, 1536, device="cuda", dtype=torch.bfloat16) * 0.02
    gate_weight = torch.randn_like(up_weight) * 0.02
    with torch.no_grad():
        actual = swiglu(x, up_weight, gate_weight)
        expected = reference_swiglu(x, up_weight, gate_weight)
    torch.testing.assert_close(actual, expected)


@skip_no_swiglu
def test_mode_one_forces_supported_shape(monkeypatch):
    monkeypatch.setenv("ASTRAI_SWIGLU", "1")
    x = torch.randn(4, 1536, device="cuda", dtype=torch.bfloat16) * 0.1
    up_weight = torch.randn(6912, 1536, device="cuda", dtype=torch.bfloat16) * 0.02
    gate_weight = torch.randn_like(up_weight) * 0.02
    with torch.no_grad():
        actual = swiglu(x, up_weight, gate_weight)
        expected = reference_swiglu(x, up_weight, gate_weight)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)


@skip_no_swiglu
def test_auto_uses_unfused_chain_until_shape_is_qualified(monkeypatch):
    # The fusion table is empty, so auto keeps the unfused linear-backend
    # chain. The linear backend may still dispatch its own GEMV for M=4,
    # hence the relaxed tolerance versus the pure-torch reference.
    monkeypatch.setenv("ASTRAI_SWIGLU", "auto")
    x = torch.randn(4, 1536, device="cuda", dtype=torch.bfloat16)
    up_weight = torch.randn(6912, 1536, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.randn_like(up_weight)
    with torch.no_grad():
        actual = swiglu(x, up_weight, gate_weight)
        expected = reference_swiglu(x, up_weight, gate_weight)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.1)


@skip_no_swiglu
def test_mode_one_falls_back_for_misaligned_storage(monkeypatch):
    """Contiguous-but-offset views must route to the unfused torch chain
    even with ASTRAI_SWIGLU=1 instead of reaching the uint4-only kernel
    (regression: the fused primitive faulted with a misaligned-address
    CUDA error for such inputs)."""
    monkeypatch.setenv("ASTRAI_SWIGLU", "1")
    k = 1536
    x_base = torch.randn(2 * k + 8, device="cuda", dtype=torch.bfloat16) * 0.1
    x = x_base[1 : 1 + 2 * k].view(2, k)
    assert x.is_contiguous() and (x.data_ptr() & 15) != 0
    up_weight = torch.randn(64, k, device="cuda", dtype=torch.bfloat16) * 0.02
    gate_weight = torch.randn_like(up_weight)
    with torch.no_grad():
        actual = swiglu(x, up_weight, gate_weight)
        expected = reference_swiglu(x, up_weight, gate_weight)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)
