import pytest
import torch
import torch.nn.functional as F

from astrai.extension import bf16_swiglu, is_available

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


@skip_no_swiglu
@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("n,k", [(6912, 1536), (4096, 4096), (11008, 4096)])
def test_bf16_swiglu_matches_common_dense_mlp_shapes(m, n, k):
    torch.manual_seed(37 + m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    up_weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * (k**-0.5)
    gate_weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * (k**-0.5)
    actual = bf16_swiglu(x, up_weight, gate_weight)
    expected = reference_swiglu(x, up_weight, gate_weight)
    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)


@skip_no_swiglu
def test_bf16_swiglu_preserves_vector_shape():
    x = torch.randn(1536, device="cuda", dtype=torch.bfloat16) * 0.1
    up_weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16) * 0.02
    gate_weight = torch.randn_like(up_weight) * 0.02
    actual = bf16_swiglu(x, up_weight, gate_weight)
    expected = reference_swiglu(x, up_weight, gate_weight)
    assert actual.shape == (256,)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)


@skip_no_swiglu
def test_bf16_swiglu_uses_current_stream_and_cuda_graph():
    torch.manual_seed(43)
    x = torch.randn(4, 1536, device="cuda", dtype=torch.bfloat16) * 0.1
    up_weight = torch.randn(6912, 1536, device="cuda", dtype=torch.bfloat16) * 0.02
    gate_weight = torch.randn_like(up_weight) * 0.02
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            bf16_swiglu(x, up_weight, gate_weight)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = bf16_swiglu(x, up_weight, gate_weight)
        x.copy_(torch.randn_like(x) * 0.1)
        graph.replay()
    stream.synchronize()
    expected = reference_swiglu(x, up_weight, gate_weight)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)


@skip_no_swiglu
@pytest.mark.parametrize(
    "make_args,error",
    [
        (
            lambda: (
                torch.randn(9, 16, device="cuda", dtype=torch.bfloat16),
                torch.randn(8, 16, device="cuda", dtype=torch.bfloat16),
                torch.randn(8, 16, device="cuda", dtype=torch.bfloat16),
            ),
            "M must",
        ),
        (
            lambda: (
                torch.randn(2, 15, device="cuda", dtype=torch.bfloat16),
                torch.randn(8, 15, device="cuda", dtype=torch.bfloat16),
                torch.randn(8, 15, device="cuda", dtype=torch.bfloat16),
            ),
            "divisible by 8",
        ),
        (
            lambda: (
                torch.randn(2, 16, device="cuda", dtype=torch.bfloat16),
                torch.randn(8, 16, device="cuda", dtype=torch.bfloat16),
                torch.randn(7, 16, device="cuda", dtype=torch.bfloat16),
            ),
            "identical shapes",
        ),
    ],
)
def test_bf16_swiglu_rejects_unsupported_inputs(make_args, error):
    with pytest.raises(RuntimeError, match=error):
        bf16_swiglu(*make_args())
