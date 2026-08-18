"""Fused BF16-boundary FP8 MMA kernel tests."""

import pytest
import torch

from astrai.extension.loader import get_module, is_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < (8, 9)
    or not is_available("fp8_mm"),
    reason="fused FP8 MMA requires a built kernel and compute capability 8.9+",
)


def _scale(tensor):
    return (tensor.abs().amax().float() / 448.0).clamp_min(1e-12)


def _quantize(tensor, scale):
    return (tensor.float() / scale).to(torch.float8_e4m3fn).float()


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(16, 8, 32), (17, 9, 33), (31, 15, 64), (32, 48, 96)],
)
def test_fused_fp8_mma_matches_explicit_quantization(m, n, k):
    torch.manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    scale_a = _scale(a)
    scale_b = _scale(b)

    out = get_module("fp8_mm").fp8_mm(a, b, scale_a, scale_b)
    expected = (
        _quantize(a, scale_a) @ _quantize(b, scale_b).t() * scale_a * scale_b
    ).to(torch.bfloat16)

    assert out.dtype == torch.bfloat16
    assert out.shape == (m, n)
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)


def test_fused_fp8_linear_forward_and_backward():
    torch.manual_seed(7)
    m, n, k = 19, 13, 37
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    scale_x, scale_w, scale_g = _scale(x), _scale(weight), _scale(grad)
    amax_x = torch.empty(1, device="cuda", dtype=torch.float32)
    amax_w = torch.empty(1, device="cuda", dtype=torch.float32)
    amax_g = torch.empty(1, device="cuda", dtype=torch.float32)
    module = get_module("fp8_mm")

    out = module.fp8_linear_forward_scaled(
        x,
        weight,
        bias,
        scale_x,
        scale_w,
        scale_x.reciprocal(),
        scale_w.reciprocal(),
        amax_x,
        amax_w,
    )
    grad_x, grad_w, grad_b = module.fp8_linear_backward_scaled(
        grad,
        x,
        weight,
        [1, 1, 1],
        scale_g,
        scale_w,
        scale_x,
        scale_g.reciprocal(),
        scale_w.reciprocal(),
        scale_x.reciprocal(),
        amax_g,
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
