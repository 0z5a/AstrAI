"""FP8 primitives: kernel-level (CUDA) and policy-level (CPU-verifiable) tests.

The kernel-level tests exercise the pure FP8 path (quantize_bf16 + mm_fp8 for
the forward GEMM, quantize + pre-quantized GEMMs for the backward); the
policy-level tests (recipes, autocast context, per-tensor meta, CPU fallbacks
of the custom ops) run without a GPU.
"""

import pytest
import torch

from astrai.extension.fp8 import (
    DelayedScaling,
    DynamicScaling,
    FP8Format,
    FP8TensorMeta,
    fp8_autocast,
    fp8_state,
)
from astrai.extension.ops.fp8 import (
    linear_backward_fp8,
    linear_forward_fp8,
    mm_fp8,
    quantize_bf16,
)
from tests.conftest import skip_no_fp8


def _scale(tensor):
    return (tensor.abs().amax().float() / 448.0).clamp_min(1e-12)


def _quantize(tensor, scale):
    return (tensor.float() / scale).to(torch.float8_e4m3fn).float()


# --------------------------------------------------------------------------
# Kernel-level (CUDA)
# --------------------------------------------------------------------------


@skip_no_fp8
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(16, 8, 32), (17, 9, 33), (31, 15, 64), (32, 48, 96)],
)
def test_fp8_mm_matches_explicit_quantization(m, n, k):
    torch.manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    scale_a = _scale(a)
    scale_b = _scale(b)
    a8, _ = quantize_bf16(a, scale_a, "e4m3")
    b8, _ = quantize_bf16(b, scale_b, "e4m3")
    out = mm_fp8(a8, b8, scale_a, scale_b)
    expected = (_quantize(a, scale_a) @ _quantize(b, scale_b) * scale_a * scale_b).to(
        torch.bfloat16
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == (m, n)
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)


@skip_no_fp8
def test_quantize_bf16_returns_amax():
    """quantize_bf16 returns (x8, amax); amax tracks the *raw* values and the
    caller never clears it (zero-initialized inside the kernel entry)."""
    torch.manual_seed(3)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.5], device="cuda")
    x8, amax = quantize_bf16(x, scale, "e4m3")
    assert x8.dtype == torch.float8_e4m3fn
    assert x8.shape == x.shape
    assert amax.shape == (1,)
    torch.testing.assert_close(amax, x.abs().amax().float().reshape(1))
    ref = (x.float() / 0.5).to(torch.float8_e4m3fn)
    assert torch.equal(x8, ref)


@skip_no_fp8
def test_quantize_bf16_e5m2_format():
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    x8, amax = quantize_bf16(x, torch.tensor([0.1], device="cuda"), "e5m2")
    assert x8.dtype == torch.float8_e5m2
    torch.testing.assert_close(amax, x.abs().amax().float().reshape(1))


@skip_no_fp8
def test_fp8_linear_forward_and_backward():
    torch.manual_seed(7)
    m, n, k = 19, 13, 37
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    scale_x, scale_w, scale_g = _scale(x), _scale(weight), _scale(grad)

    out, amax_x, amax_w = linear_forward_fp8(x, weight, bias, scale_x, scale_w)
    grad_x, grad_w, grad_b, amax_g = linear_backward_fp8(
        grad, x, weight, [1, 1, 1], scale_g, scale_w, scale_x, "e4m3"
    )

    qx = _quantize(x, scale_x)
    qw = _quantize(weight, scale_w)
    qg = _quantize(grad, scale_g)
    expected_out = (qx @ qw.t() * scale_x * scale_w + bias).to(torch.bfloat16)
    expected_grad_x = (qg @ qw * scale_g * scale_w).to(torch.bfloat16)
    expected_grad_w = (qg.t() @ qx * scale_g * scale_x).to(torch.bfloat16)

    torch.testing.assert_close(out, expected_out, atol=0.125, rtol=0.01)
    torch.testing.assert_close(grad_x, expected_grad_x, atol=0.125, rtol=0.01)
    torch.testing.assert_close(grad_w, expected_grad_w, atol=0.125, rtol=0.01)
    torch.testing.assert_close(grad_b, grad.sum(0).to(torch.bfloat16))
    torch.testing.assert_close(amax_x, x.abs().amax().float().reshape(1))
    torch.testing.assert_close(amax_w, weight.abs().amax().float().reshape(1))
    torch.testing.assert_close(amax_g, grad.abs().amax().float().reshape(1))


@skip_no_fp8
def test_linear_backward_e5m2_gradients():
    """Hybrid backward: gradient GEMMs run in E5M2 (larger dynamic range)."""
    torch.manual_seed(5)
    m, n, k = 32, 16, 64
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 3.0
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 10.0
    sg = _scale(grad) * 0.5
    sw = _scale(weight)
    sx = _scale(x)

    grad_x, grad_w, grad_b, amax_g = linear_backward_fp8(
        grad, x, weight, [1, 1, 1], sg, sw, sx, "e5m2"
    )

    def q5(t, s):
        return (t.float() / s).to(torch.float8_e5m2).float()

    qg = q5(grad, sg)
    qw = q5(weight, sw)
    qx = q5(x, sx)
    expected_grad_x = (qg @ qw * sg * sw).to(torch.bfloat16)
    expected_grad_w = (qg.t() @ qx * sg * sx).to(torch.bfloat16)
    torch.testing.assert_close(grad_x, expected_grad_x, atol=0.5, rtol=0.05)
    torch.testing.assert_close(grad_w, expected_grad_w, atol=0.5, rtol=0.05)
    torch.testing.assert_close(amax_g, grad.abs().amax().float().reshape(1))


@skip_no_fp8
def test_fp8_linear_static_fp8_weight_and_bias():
    """Static fp8 inference: pre-quantized w8/b8 + their scales take the GEMM
    directly (no weight quantize, amax_w = 0); the bias is fused in the
    epilogue (bf16 and fp8 bias share the fused path)."""
    torch.manual_seed(9)
    m, n, k = 67, 45, 129
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.5
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.5
    sx, sw, sb = _scale(x), _scale(weight), _scale(bias)

    w8, _ = quantize_bf16(weight, sw, "e4m3")
    b8, _ = quantize_bf16(bias, sb, "e4m3")
    out, amax_x, amax_w = linear_forward_fp8(x, w8, b8, sx, sw, "e4m3", sb)

    qx = _quantize(x, sx)
    qw = _quantize(weight, sw)
    qb = _quantize(bias, sb)
    expected = (qx @ qw.t() * sx * sw + qb * sb).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)
    torch.testing.assert_close(amax_x, x.abs().amax().float().reshape(1))
    assert amax_w.item() == 0.0  # nothing measured on the static path

    # bf16 bias stays bf16 on the same fused-epilogue path
    out_bf16bias, _, _ = linear_forward_fp8(x, w8, bias, sx, sw, "e4m3")
    expected_b = (qx @ qw.t() * sx * sw + bias.float()).to(torch.bfloat16)
    torch.testing.assert_close(out_bf16bias, expected_b, atol=0.125, rtol=0.01)


@skip_no_fp8
def test_fp8_linear_backward_outside_autocast():
    """aten::linear records an fp8 autograd node inside fp8_autocast; the
    backward runs fp8 kernels even after the context exits (loss.backward()
    placement is free), instead of falling back to bf16 mm."""
    import torch.nn.functional as F

    import astrai.extension.fp8 as f8mod

    torch.manual_seed(5)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        96, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr, wr, br = (t.detach().clone().requires_grad_() for t in (x, weight, bias))

    calls = {"bwd": 0}
    orig = f8mod.linear_backward_fp8

    def spy(g, xx, ww, masks, sg, sw, sx, fmt="e5m2"):
        calls["bwd"] += 1
        return orig(g, xx, ww, masks, sg, sw, sx, fmt)

    f8mod.linear_backward_fp8 = spy
    try:
        with fp8_autocast(enabled=True):
            out = F.linear(x, weight, bias)
        assert type(out.grad_fn).__name__ == "_LinearFp8Backward"
        out.float().pow(2).sum().backward()  # outside the autocast region
    finally:
        f8mod.linear_backward_fp8 = orig
        f8mod.fp8_state().reset()

    assert calls["bwd"] == 1  # fp8 kernels, not the bf16 fallback
    ref = F.linear(xr, wr, br)
    ref.float().pow(2).sum().backward()

    # E5M2 backward quantization noise: compare directions/norms (the
    # torchao/TE style) rather than elementwise against the bf16 reference.
    def _direction(a, b):
        cos = torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        )
        return cos > 0.99 and 0.9 < a.float().norm() / b.float().norm() < 1.1

    assert _direction(x.grad, xr.grad)
    assert _direction(weight.grad, wr.grad)
    assert _direction(bias.grad, br.grad)


@skip_no_fp8
def test_mm_fp8_matches_scaled_mm():
    torch.manual_seed(11)
    m, n, k = 512, 4096, 4096
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    sa = torch.tensor([2.5], device="cuda")
    sb = torch.tensor([1.5], device="cuda")
    a8, _ = quantize_bf16(a, sa, "e4m3")
    b8, _ = quantize_bf16(b, sb, "e4m3")
    out = mm_fp8(a8, b8, sa, sb)
    assert out.dtype == torch.bfloat16
    assert out.shape == (m, n)

    ref = (a8.float().double() @ b8.float().double() * 2.5 * 1.5).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=6.0, rtol=0.05)

    try:
        torch._scaled_mm(a8, b8, sa, sb, out_dtype=torch.bfloat16)
    except (RuntimeError, NotImplementedError):
        return
    torch.testing.assert_close(
        out,
        torch._scaled_mm(a8, b8, sa, sb, out_dtype=torch.bfloat16),
        atol=2.0,
        rtol=0.01,
    )


@skip_no_fp8
def test_mm_fp8_fp8_output():
    """mm_fp8 with out_dtype='e4m3' produces an FP8 output (layer-to-layer)."""
    torch.manual_seed(12)
    m, n, k = 256, 128, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    sa = torch.tensor([2.0], device="cuda")
    sb = torch.tensor([1.0], device="cuda")
    os_ = torch.tensor([0.5], device="cuda")
    a8, _ = quantize_bf16(a, sa, "e4m3")
    b8, _ = quantize_bf16(b, sb, "e4m3")
    out8 = mm_fp8(a8, b8, sa, sb, out_dtype="e4m3", out_scale=os_)
    assert out8.dtype == torch.float8_e4m3fn
    assert out8.shape == (m, n)

    ref = (a8.float().double() @ b8.float().double() * 2.0 * 1.0 * 0.5).to(
        torch.bfloat16
    )
    torch.testing.assert_close(
        out8.float().to(torch.bfloat16), ref, atol=6.0, rtol=0.05
    )


# --------------------------------------------------------------------------
# Policy-level (CPU-verifiable)
# --------------------------------------------------------------------------


def test_recipe_scale_from_history():
    """Delayed: max over the window + margin; dynamic: current amax."""
    hist = torch.tensor([1.0, 2.0, 0.5])
    d = DelayedScaling(history_len=3, margin=0)
    assert torch.allclose(d.scale_from_history(hist, "e4m3"), torch.tensor(2.0 / 448.0))
    d_m = DelayedScaling(history_len=3, margin=2)
    assert torch.allclose(
        d_m.scale_from_history(hist, "e4m3"), torch.tensor(2.0 / 448.0 / 4.0)
    )
    dyn = DynamicScaling()
    amax = torch.tensor([0.25])
    assert torch.allclose(
        dyn.scale_from_history(amax, "e4m3"), torch.tensor(0.25 / 448.0)
    )
    assert torch.allclose(
        dyn.scale_from_history(amax, "e5m2"), torch.tensor(0.25 / 57344.0)
    )


def test_fp8_format_enum():
    assert FP8Format.HYBRID.fwd() == "e4m3"
    assert FP8Format.HYBRID.bwd() == "e5m2"
    assert FP8Format.E4M3.fwd() == FP8Format.E4M3.bwd() == "e4m3"
    assert FP8Format.E5M2.fwd() == FP8Format.E5M2.bwd() == "e5m2"


def test_fp8_autocast_context():
    """fp8_autocast sets and restores recipe + format on the global state."""
    state = fp8_state()
    prev = (state.enabled, state.recipe, state.fp8_format)
    try:
        with fp8_autocast(enabled=True, fp8_format="hybrid", update_interval=8):
            assert state.enabled
            assert isinstance(state.recipe, DelayedScaling)
            assert state.recipe.history_len == 8
            assert state.fp8_format is FP8Format.HYBRID
            with fp8_autocast(enabled=True, recipe=DynamicScaling(), fp8_format="e4m3"):
                assert isinstance(state.recipe, DynamicScaling)
                assert state.fp8_format is FP8Format.E4M3
            assert state.fp8_format is FP8Format.HYBRID  # restored on exit
        assert not state.enabled
    finally:
        state.enabled, state.recipe, state.fp8_format = prev


def test_fp8_tensor_meta_delayed_update():
    """Meta seeds from data and refreshes the scale from the amax ring."""
    meta = FP8TensorMeta(torch.device("cpu"), DelayedScaling(history_len=4, margin=0))
    w = torch.randn(8, 8)
    meta.w.seed(w, "e4m3")
    assert meta.w.initialized
    torch.testing.assert_close(meta.w.scale, (w.abs().amax() / 448.0).reshape(1))
    meta.w.update(torch.tensor([4.0]), "e4m3")
    torch.testing.assert_close(meta.w.scale, torch.tensor(4.0 / 448.0).reshape(1))


def test_quantize_bf16_cpu_fallback():
    """CPU fallback of the quantize primitive (scale semantics + amax)."""
    x = torch.randn(16, 32, dtype=torch.bfloat16)
    scale = torch.tensor([0.5])
    x8, amax = quantize_bf16(x, scale, "e4m3")
    assert x8.dtype == torch.float8_e4m3fn
    ref = (x.float() / 0.5).to(torch.float8_e4m3fn)
    assert torch.equal(x8, ref)
    torch.testing.assert_close(amax, x.abs().amax().float().reshape(1))


def test_mm_fp8_cpu_fallback():
    a8 = torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn)
    b8 = torch.tensor([[3.0], [4.0]], dtype=torch.float8_e4m3fn)
    sa = torch.tensor([2.0])
    sb = torch.tensor([0.5])
    out = mm_fp8(a8, b8, sa, sb)
    ref = (a8.float() @ b8.float() * 2.0 * 0.5).to(torch.bfloat16)
    torch.testing.assert_close(out, ref)


def test_mm_fp8_fp8_output_cpu():
    """CPU fallback with an FP8 output (out_dtype='e4m3' + out_scale)."""
    a8 = torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn)
    b8 = torch.tensor([[3.0], [4.0]], dtype=torch.float8_e4m3fn)
    sa = torch.tensor([2.0])
    sb = torch.tensor([0.5])
    os_ = torch.tensor([0.25])
    out8 = mm_fp8(a8, b8, sa, sb, out_dtype="e4m3", out_scale=os_)
    assert out8.dtype == torch.float8_e4m3fn
    ref = (a8.float() @ b8.float() * 2.0 * 0.5 * 0.25).to(torch.float8_e4m3fn)
    assert torch.equal(out8, ref)
