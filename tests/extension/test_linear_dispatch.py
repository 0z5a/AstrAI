import importlib
import logging

import pytest
import torch
import torch.nn.functional as F

from astrai.extension import is_available, linear
from astrai.extension.backend import linear as public_linear
from astrai.extension.dispatch import explain, op_backend, resolve

# The package attribute ``linear`` is the dispatched function; reach the
# module object explicitly for monkeypatching its private helpers.
linear_module = importlib.import_module("astrai.extension.backend.linear")

GEMV_AVAILABLE = (
    torch.cuda.is_available()
    and is_available("bf16_gemv")
    and torch.cuda.get_device_capability() >= (8, 0)
)
skip_no_gemv = pytest.mark.skipif(
    not GEMV_AVAILABLE,
    reason="BF16 GEMV requires a built kernel and compute capability 8.0+",
)


def _routes_to_gemv(monkeypatch, x, weight, bias=None) -> bool:
    """Patch the GEMV entry point to a sentinel and report whether
    ``linear`` selected it (torch fallback would compute a real tensor)."""
    sentinel = object()

    def fake_gemv(x, weight, bias):
        return sentinel

    monkeypatch.setattr(linear_module, "_inference_bf16_gemv", fake_gemv)
    return linear(x, weight, bias) is sentinel


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
    from astrai.model.components.linear import Linear

    layer = Linear(3, 4)
    assert layer(torch.randn(2, 3)) is sentinel


def test_invalid_mode_warns_and_uses_auto(monkeypatch, caplog):
    monkeypatch.setenv("ASTRAI_GEMV", "invalid-test-mode")
    x = torch.randn(2, 8)
    weight = torch.randn(4, 8)
    with caplog.at_level(logging.WARNING):
        actual = linear(x, weight)
    assert "using auto" in caplog.text
    torch.testing.assert_close(actual, F.linear(x, weight))


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


@skip_no_gemv
@pytest.mark.parametrize("m", [2, 3, 4])
def test_auto_selects_small_decode_batches(monkeypatch, m):
    monkeypatch.setenv("ASTRAI_GEMV", "auto")
    x = torch.randn(m, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert _routes_to_gemv(monkeypatch, x, weight)


@skip_no_gemv
@pytest.mark.parametrize("m", [1, 5, 8, 9])
def test_auto_falls_back_outside_band(monkeypatch, m):
    monkeypatch.setenv("ASTRAI_GEMV", "auto")
    x = torch.randn(m, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert not _routes_to_gemv(monkeypatch, x, weight)
        torch.testing.assert_close(
            linear(x, weight), F.linear(x, weight), rtol=0.02, atol=0.25
        )


@skip_no_gemv
def test_mode_zero_disables_gemv(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "0")
    x = torch.randn(2, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert not _routes_to_gemv(monkeypatch, x, weight)
        torch.testing.assert_close(linear(x, weight), F.linear(x, weight))


@skip_no_gemv
@pytest.mark.parametrize("m", [1, 2, 8])
def test_mode_one_forces_every_capable_batch(monkeypatch, m):
    monkeypatch.setenv("ASTRAI_GEMV", "1")
    x = torch.randn(m, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert _routes_to_gemv(monkeypatch, x, weight)


@skip_no_gemv
def test_mode_one_rejects_oversized_batch_and_grad(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "1")
    weight = torch.randn(
        256, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    with torch.no_grad():
        oversized = torch.randn(9, 1536, device="cuda", dtype=torch.bfloat16)
        assert not _routes_to_gemv(monkeypatch, oversized, weight)
    assert not _routes_to_gemv(
        monkeypatch, torch.randn(2, 1536, device="cuda", dtype=torch.bfloat16), weight
    )


@skip_no_gemv
def test_mode_one_supports_bias_and_vector_input(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "1")
    x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert _routes_to_gemv(monkeypatch, x, weight, bias)
        monkeypatch.undo()
        torch.testing.assert_close(
            linear(x, weight, bias),
            F.linear(x, weight, bias),
            rtol=0.02,
            atol=0.25,
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


def test_linear_family_is_registered_with_shared_dispatcher():
    from astrai.extension.dispatch import _FAMILIES

    assert "linear" in _FAMILIES
    x = torch.randn(2, 8)
    weight = torch.randn(4, 8)
    resolution = resolve("linear", x, weight)
    assert resolution.record.family == "linear"
    assert resolution.origin in ("chain", "fallback")
    assert "linear" in explain("linear", x, weight)


@skip_no_gemv
def test_ops_env_override_forces_torch_for_capable_call(monkeypatch):
    """ASTR_OPS=linear=torch must keep working after the M-band rewrite
    (regression: the family was silently dropped from the dispatcher, so
    the override warned, fell through, and the gemv kernel still ran)."""
    monkeypatch.setenv("ASTR_OPS", "linear=torch")
    x = torch.randn(2, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert not _routes_to_gemv(monkeypatch, x, weight)
        torch.testing.assert_close(
            linear(x, weight), F.linear(x, weight), rtol=0.02, atol=0.25
        )


@skip_no_gemv
def test_ops_env_override_forces_gemv(monkeypatch):
    monkeypatch.setenv("ASTR_OPS", "linear=gemv")
    x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        # M=1 is outside the auto band but inside the forced gemv record.
        assert _routes_to_gemv(monkeypatch, x, weight)


@skip_no_gemv
def test_op_backend_context_selects_torch(monkeypatch):
    monkeypatch.setenv("ASTRAI_GEMV", "1")
    x = torch.randn(2, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), op_backend(linear="torch"):
        assert not _routes_to_gemv(monkeypatch, x, weight)
        torch.testing.assert_close(
            linear(x, weight), F.linear(x, weight), rtol=0.02, atol=0.25
        )
    # The override is scoped: the forced mode applies again afterwards.
    with torch.no_grad():
        assert _routes_to_gemv(monkeypatch, x, weight)


def test_op_backend_rejects_unknown_linear_handle():
    with pytest.raises(ValueError, match="Unknown linear implementation"):
        op_backend(linear="nonexistent").__enter__()
