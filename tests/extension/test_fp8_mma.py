"""FP8 primitives: kernel-level (CUDA) and policy-level (CPU-verifiable) tests.

The kernel-level tests exercise the two stateless primitives (``quantize`` for
bf16/fp16/fp32 -> FP8, ``mm_fp8`` for the pre-quantized GEMM with transposed
operands); the policy-level tests (recipes, autocast context, per-tensor meta,
CPU fallbacks of the custom ops) run without a GPU.
"""

import threading

import pytest
import torch
import torch.nn.functional as F

import astrai.extension.fp8 as f8mod
from astrai.extension.fp8 import (
    DelayedScaling,
    DynamicScaling,
    FP8Format,
    FP8TensorMeta,
    _ScaleRing,
    fp8_autocast,
    fp8_linear_enable,
    fp8_linear_enabled,
    fp8_state,
)
from astrai.extension.ops.fp8 import mm_fp8, quantize
from tests.conftest import skip_no_fp8


def _scale(tensor):
    return (tensor.abs().amax().float() / 448.0).clamp_min(1e-12)


def _quantize(tensor, scale, fmt="e4m3"):
    """Reference quantize: multiply by the reciprocal (the kernel's exact
    arithmetic — a plain divide flips fp8 boundary cases by one ulp)."""
    dtype = torch.float8_e5m2 if fmt == "e5m2" else torch.float8_e4m3fn
    return (tensor.float() * scale.reciprocal()).to(dtype).float()


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
    a8, _ = quantize(a, scale_a.reciprocal(), "e4m3")
    b8, _ = quantize(b, scale_b.reciprocal(), "e4m3")
    out = mm_fp8(a8, b8, scale_a * scale_b)
    expected = (_quantize(a, scale_a) @ _quantize(b, scale_b) * scale_a * scale_b).to(
        torch.bfloat16
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == (m, n)
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)


@skip_no_fp8
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_quantize_input_dtypes(in_dtype, fmt):
    """quantize accepts bf16/fp16/fp32 inputs; bytes and amax match the
    explicit (value * multiplier) reference."""
    torch.manual_seed(3)
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32) * 0.5
    x = x.to(in_dtype)
    scale = torch.tensor([0.5], device="cuda")
    x8, amax = quantize(x, scale, fmt)
    out_dtype = torch.float8_e5m2 if fmt == "e5m2" else torch.float8_e4m3fn
    assert x8.dtype == out_dtype
    assert x8.shape == x.shape
    assert amax.shape == (1,)
    torch.testing.assert_close(amax, x.abs().amax().float().reshape(1))
    ref = (x.float() * 0.5).to(out_dtype)
    assert torch.equal(x8, ref)


@skip_no_fp8
def test_quantize_e5m2_format():
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    x8, amax = quantize(x, torch.tensor([10.0], device="cuda"), "e5m2")
    assert x8.dtype == torch.float8_e5m2
    torch.testing.assert_close(amax, x.abs().amax().float().reshape(1))


@skip_no_fp8
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
def test_mm_fp8_transposed_operands(trans_a, trans_b):
    """mm_fp8 handles all four operand layouts via trans_a/trans_b."""
    torch.manual_seed(17)
    m, n, k = 19, 13, 37
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)  # A [M][K]
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)  # B^T [N][K]
    sa, sb = _scale(a), _scale(b)
    a8, _ = quantize(a, sa.reciprocal(), "e4m3")
    b8, _ = quantize(b, sb.reciprocal(), "e4m3")
    a_op = a8.t().contiguous() if trans_a else a8
    b_op = b8 if trans_b else b8.t().contiguous()

    out = mm_fp8(a_op, b_op, sa * sb, trans_a=trans_a, trans_b=trans_b)
    assert out.shape == (m, n)
    expected = (_quantize(a, sa) @ _quantize(b, sb).t() * sa * sb).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)


@skip_no_fp8
def test_delayed_scaling_forward_uses_snapshot_scale():
    """The delayed scale for step N is computed from amax(steps < N); the
    forward must snapshot the scale before the ring update, so a changing
    amax across steps does not leak the next-step scale into the output."""
    torch.manual_seed(11)
    dev = torch.device("cuda")
    state = f8mod.fp8_state()
    state.reset()
    state.default_recipe = DelayedScaling(history_len=1, margin=0)
    state.default_format = FP8Format.E4M3
    try:
        m, n, k = 32, 16, 64
        x1 = torch.randn(m, k, device=dev, dtype=torch.bfloat16) * 0.5
        # Smaller amax than x1: the delayed scale (amax(x1)/448) still covers
        # x2 without fp8 saturation, while the next-step scale would differ.
        x2 = torch.randn(m, k, device=dev, dtype=torch.bfloat16) * 0.35
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.5
        bias = torch.zeros(n, device=dev, dtype=torch.bfloat16)

        f8mod.fp8_linear_forward(x1, w, bias)  # step 1: seeds the rings
        out2, _, _ = f8mod.fp8_linear_forward(x2, w, bias)  # amax changes
        torch.cuda.synchronize()

        # The delayed scale for step 2 is amax(x1)/448 (history_len=1); the
        # GEMM must use that same scale for dequant as the quantize used.
        sx = _scale(x1)
        sw = _scale(w)
        qx = _quantize(x2, sx)
        qw = _quantize(w, sw)
        expected = (qx @ qw.t() * sx * sw + bias).to(torch.bfloat16)
        torch.testing.assert_close(out2, expected, atol=0.125, rtol=0.01)
    finally:
        state.reset()


@skip_no_fp8
def test_fp8_linear_forward_and_backward():
    """The composed strategy path: forward quantize+GEMM+bias, backward
    dX/dW GEMMs on transposed operands (E5M2 in hybrid)."""
    torch.manual_seed(7)
    m, n, k = 19, 13, 37
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    state = f8mod.fp8_state()
    state.reset()
    state.default_recipe = DynamicScaling()
    try:
        out, _, _ = f8mod.fp8_linear_forward(x, weight, bias)

        sx, sw = _scale(x), _scale(weight)
        qx = _quantize(x, sx)
        qw = _quantize(weight, sw)
        expected_out = (qx @ qw.t() * sx * sw + bias).to(torch.bfloat16)
        torch.testing.assert_close(out, expected_out, atol=0.125, rtol=0.01)

        # backward through the aten::linear integration (hybrid E5M2). The
        # incoming gradient is 2*out of the *fp8* forward (bf16-rounded), not
        # 2*exact — derive the reference from the actual output.
        xr = x.detach().clone().requires_grad_()
        wr = weight.detach().clone().requires_grad_()
        br = bias.detach().clone().requires_grad_()
        with fp8_autocast(enabled=True):
            loss = F.linear(xr, wr, br).float().pow(2).sum()
        loss.backward()

        g = (2 * out.float()).to(torch.bfloat16).float()  # actual grad wrt out
        # the dynamic path measures current-step amax in the bwd fmt (E5M2);
        # amax must be taken in fp32 — a bf16-rounded scale flips E5M2
        # boundary rounding (2-bit mantissa) and the reference drifts.
        e5 = 57344.0
        sg = (g.abs().amax() / e5).clamp_min(1e-12)
        sw5 = (weight.abs().amax().float() / e5).clamp_min(1e-12)
        sx5 = (x.abs().amax().float() / e5).clamp_min(1e-12)
        expected_grad_x = (
            _quantize(g, sg, "e5m2") @ _quantize(weight, sw5, "e5m2") * sg * sw5
        ).to(torch.bfloat16)
        expected_grad_w = (
            _quantize(g, sg, "e5m2").t() @ _quantize(x, sx5, "e5m2") * sg * sx5
        ).to(torch.bfloat16)
        torch.testing.assert_close(xr.grad, expected_grad_x, atol=0.5, rtol=0.05)
        torch.testing.assert_close(wr.grad, expected_grad_w, atol=0.5, rtol=0.05)
        torch.testing.assert_close(
            br.grad, g.sum(0).to(torch.bfloat16), atol=0.5, rtol=0.05
        )
    finally:
        state.reset()


@skip_no_fp8
def test_fp8_linear_backward_outside_autocast():
    """aten::linear records an fp8 autograd node inside fp8_autocast; the
    backward runs fp8 kernels even after the context exits (loss.backward()
    placement is free), instead of falling back to bf16 mm."""
    torch.manual_seed(5)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        96, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr, wr, br = (t.detach().clone().requires_grad_() for t in (x, weight, bias))

    calls = {"fwd": 0}
    orig = f8mod.fp8_linear_forward

    def spy(*args, **kwargs):
        calls["fwd"] += 1
        return orig(*args, **kwargs)

    f8mod.fp8_linear_forward = spy
    try:
        with fp8_autocast(enabled=True):
            out = F.linear(x, weight, bias)
        assert type(out.grad_fn).__name__ == "_LinearFp8Backward"
        out.float().pow(2).sum().backward()  # outside the autocast region
    finally:
        f8mod.fp8_linear_forward = orig
        f8mod.fp8_state().reset()

    assert calls["fwd"] == 1  # fp8 kernels, not the bf16 fallback
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
    sa = _scale(a)
    sb = _scale(b)
    a8, _ = quantize(a, sa.reciprocal(), "e4m3")
    b8, _ = quantize(b, sb.reciprocal(), "e4m3")
    out = mm_fp8(a8, b8, sa * sb)
    assert out.dtype == torch.bfloat16
    assert out.shape == (m, n)

    ref = (a8.float().double() @ b8.float().double() * sa * sb).to(torch.bfloat16)
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
    """Meta seeds from data; hist/scale are packed views of one state buffer."""
    meta = FP8TensorMeta(torch.device("cpu"), DelayedScaling(history_len=4, margin=0))
    w = torch.randn(8, 8)
    meta.w.seed(w, "e4m3")
    assert meta.w.initialized
    torch.testing.assert_close(meta.w.scale, (w.abs().amax() / 448.0).reshape(1))
    # [hist | scale] packing: views alias the single state buffer.
    assert meta.w.state.numel() == 4 + 2
    assert meta.w.hist.data_ptr() == meta.w.state.data_ptr()
    assert meta.w.scale.data_ptr() == meta.w.state[4:].data_ptr()
    meta.w.advance()
    assert meta.w.idx == 1

    # update folds a fresh amax into the window and publishes the next scale
    amax = torch.tensor([8.0])
    meta.w.update(amax, "e4m3")
    torch.testing.assert_close(meta.w.scale, torch.tensor([8.0 / 448.0]))


def test_quantize_cpu_fallback():
    """CPU fallback of the quantize primitive (scale semantics + amax)."""
    x = torch.randn(16, 32, dtype=torch.bfloat16)
    scale = torch.tensor([0.5])  # quantize multiplier
    x8, amax = quantize(x, scale, "e4m3")
    assert x8.dtype == torch.float8_e4m3fn
    ref = (x.float() * 0.5).to(torch.float8_e4m3fn)
    assert torch.equal(x8, ref)
    torch.testing.assert_close(amax, x.abs().amax().float().reshape(1))


def test_mm_fp8_cpu_fallback():
    a8 = torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn)
    b8 = torch.tensor([[3.0], [4.0]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([1.0])
    out = mm_fp8(a8, b8, scale)
    ref = (a8.float() @ b8.float() * 1.0).to(torch.bfloat16)
    torch.testing.assert_close(out, ref)


# --------------------------------------------------------------------------
# torch-autocast parity: context semantics (nesting, thread locality, switch)
# --------------------------------------------------------------------------


def _linear():
    """Shared helper: a small bf16 linear operand set on CUDA (grad-tracking
    so aten::linear records an autograd node)."""
    torch.manual_seed(31)
    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return x, w


@skip_no_fp8
def test_nested_disabled_region_redispatches_bf16():
    """A nested fp8_autocast(enabled=False) region temporarily restores the
    bf16 aten::linear path (torch's nested-disable semantics), and fp8
    resumes when it exits."""
    x, w = _linear()
    with fp8_autocast(enabled=True):
        F.linear(x, w)
        with fp8_autocast(enabled=False):
            out_bf16 = F.linear(x, w)
            assert type(out_bf16.grad_fn).__name__ != "_LinearFp8Backward"
            assert out_bf16.dtype == torch.bfloat16
        out_again = F.linear(x, w)
        assert type(out_again.grad_fn).__name__ == "_LinearFp8Backward"


@skip_no_fp8
def test_global_switch_routes_without_region():
    """fp8_linear_enable(True) routes aten::linear to fp8 outside any region
    (the persistent default); disabling restores bf16."""
    x, w = _linear()
    state = fp8_state()
    try:
        fp8_linear_enable(True)
        out = F.linear(x, w)
        assert type(out.grad_fn).__name__ == "_LinearFp8Backward"
        fp8_linear_enable(False)
        out = F.linear(x, w)
        assert type(out.grad_fn).__name__ != "_LinearFp8Backward"
    finally:
        state.reset()


def test_autocast_state_is_thread_local():
    """torch parity: the active config is thread-local — another thread does
    not see an open region (CPU-only check of the flag, no kernels)."""
    seen = {}
    with fp8_autocast(enabled=True):
        assert fp8_linear_enabled()
        t = threading.Thread(target=lambda: seen.update(enabled=fp8_linear_enabled()))
        t.start()
        t.join()
    assert seen["enabled"] is False
    assert not fp8_linear_enabled()
