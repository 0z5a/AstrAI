import logging

import pytest
import torch
import torch.nn.functional as F

from astrai.extension import explain, is_available, linear, op_backend
from astrai.extension.backend import linear as public_linear
from astrai.model.components.linear import Linear

GEMV_AVAILABLE = (
    torch.cuda.is_available()
    and is_available("bf16_gemv")
    and torch.cuda.get_device_capability() >= (8, 0)
)
skip_no_gemv = pytest.mark.skipif(
    not GEMV_AVAILABLE,
    reason="BF16 GEMV requires a built kernel and compute capability 8.0+",
)


def test_linear_backend_is_public():
    assert linear is public_linear


def test_model_linear_routes_through_backend(monkeypatch):
    sentinel = torch.randn(2, 4)

    def fake_linear(x, weight, bias):
        assert x.shape == (2, 3)
        assert weight.shape == (4, 3)
        assert bias is None
        return sentinel

    monkeypatch.setattr("astrai.model.components.linear.linear", fake_linear)
    layer = Linear(3, 4)
    assert layer(torch.randn(2, 3)) is sentinel


def test_cpu_and_training_calls_fall_back_to_torch(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "1")
    x = torch.randn(2, 8, requires_grad=True)
    weight = torch.randn(4, 8, requires_grad=True)
    actual = linear(x, weight)
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None
    assert weight.grad is not None


def test_invalid_mode_warns_and_uses_auto(monkeypatch, caplog):
    monkeypatch.setenv("ASTRAI_GEMV", "invalid-test-mode")
    with caplog.at_level(logging.WARNING):
        trace = explain("linear", torch.randn(1, 8), torch.randn(4, 8))
    assert "using auto" in caplog.text
    assert "=> torch" in trace


@skip_no_gemv
def test_mode_zero_disables_gemv(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "0")
    x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert "=> torch" in explain("linear", x, weight)
        torch.testing.assert_close(linear(x, weight), F.linear(x, weight))


@skip_no_gemv
def test_mode_one_forces_capable_unmeasured_shape(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "1")
    x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert "=> gemv" in explain("linear", x, weight)
        torch.testing.assert_close(
            linear(x, weight), F.linear(x, weight), rtol=0.02, atol=0.25
        )


@skip_no_gemv
@pytest.mark.parametrize("m", [2, 4, 8])
def test_mode_one_dispatches_supported_small_batches(monkeypatch, m):
    monkeypatch.setenv("ASTRAI_GEMV", "1")
    x = torch.randn(m, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert "=> gemv" in explain("linear", x, weight)
        torch.testing.assert_close(
            linear(x, weight), F.linear(x, weight), rtol=0.02, atol=0.25
        )


@skip_no_gemv
def test_auto_m1_falls_back_until_end_to_end_gate_passes(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "auto")
    x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    winning = torch.randn(
        1536,
        1536,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with torch.no_grad():
        assert "=> torch" in explain("linear", x, winning)
        torch.testing.assert_close(linear(x, winning), F.linear(x, winning))


@skip_no_gemv
def test_auto_selects_measured_sm89_small_batch_winner(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "auto")
    x = torch.randn(4, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        trace = explain("linear", x, weight)
        if torch.cuda.get_device_capability() == (8, 9):
            assert "=> auto_gemv" in trace
        else:
            assert "=> torch" in trace
        torch.testing.assert_close(
            linear(x, weight), F.linear(x, weight), rtol=0.02, atol=0.5
        )


@skip_no_gemv
@pytest.mark.parametrize(
    "m,n,k",
    [
        (2, 6912, 1536),  # up/gate loses at every measured M
        (2, 1536, 6912),  # long-K accumulation changed checkpoint greedy output
        (4, 100000, 1536),  # LM head misses the 5% M=4 gate
        (4, 1536, 6912),  # long-K accumulation changed checkpoint greedy output
        (8, 256, 1536),  # remaining M=8 winners miss the 3% end-to-end gate
    ],
)
def test_auto_rejects_measured_small_batch_losers(monkeypatch, m, n, k):
    monkeypatch.setenv("ASTRAI_GEMV", "auto")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert "=> torch" in explain("linear", x, weight)


@skip_no_gemv
def test_grad_enabled_and_unsupported_multirow_always_fall_back(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "1")
    x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        256, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    assert "=> torch" in explain("linear", x, weight)
    with torch.no_grad():
        oversized = torch.randn(9, 1536, device="cuda", dtype=torch.bfloat16)
        assert "=> torch" in explain("linear", oversized, weight)


@skip_no_gemv
def test_explicit_gemv_selection_respects_capability(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "0")
    x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), op_backend(linear="gemv"):
        assert "=> gemv" in explain("linear", x, weight)
        torch.testing.assert_close(
            linear(x, weight), F.linear(x, weight), rtol=0.02, atol=0.25
        )


@skip_no_gemv
def test_dispatched_linear_cuda_graph_replay(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "1")
    x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(3):
            linear(x, weight)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = linear(x, weight)
        x.copy_(torch.randn_like(x))
        graph.replay()
        expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)
