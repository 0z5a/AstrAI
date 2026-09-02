import pytest
import torch
import torch.nn.functional as F

from astrai.extension import bf16_gemv, is_available

GEMV_AVAILABLE = (
    torch.cuda.is_available()
    and is_available("bf16_gemv")
    and torch.cuda.get_device_capability() >= (8, 0)
)
skip_no_gemv = pytest.mark.skipif(
    not GEMV_AVAILABLE,
    reason="BF16 GEMV requires a built kernel and compute capability 8.0+",
)


@skip_no_gemv
@pytest.mark.parametrize(
    "n,k",
    [(256, 1536), (1536, 1536), (6912, 1536), (1536, 6912), (100000, 1536)],
)
def test_bf16_gemv_matches_linear_shape_families(n, k):
    torch.manual_seed(17)
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemv(x, weight)
    expected = F.linear(x, weight)
    assert actual.shape == (n,)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemv
@pytest.mark.parametrize("m", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("n,k", [(256, 1536), (1536, 1536), (1536, 6912)])
def test_bf16_gemv_matches_small_decode_batches(m, n, k):
    torch.manual_seed(19 + m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemv(x, weight)
    expected = F.linear(x, weight)
    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.5)


@skip_no_gemv
@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize(
    "n,k",
    [
        (1024, 4096),
        (4096, 4096),
        (11008, 4096),
        (4096, 11008),
        (14336, 4096),
        (4096, 14336),
        (5120, 5120),
        (13824, 5120),
        (5120, 13824),
        (16384, 4096),
        (4096, 16384),
        (512, 3584),
        (3584, 3584),
        (18944, 3584),
        (3584, 18944),
        (1024, 8192),
        (8192, 8192),
        (28672, 8192),
        (8192, 28672),
        (2048, 2048),
        (8192, 2048),
        (2048, 8192),
    ],
)
def test_bf16_gemv_matches_common_transformer_shapes(m, n, k):
    torch.manual_seed(2026 + m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    weight.normal_(mean=0.0, std=0.02)
    actual = bf16_gemv(x, weight)
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemv
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 8192, 2048),
        (8, 4096, 11008),
        (8, 512, 3584),
        (8, 1024, 8192),
        (8, 2048, 8192),
    ],
)
def test_bf16_gemv_matches_m8_edge_bands(m, n, k):
    torch.manual_seed(2026 + m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    weight.normal_(mean=0.0, std=0.02)
    actual = bf16_gemv(x, weight)
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemv
def test_bf16_gemv_preserves_singleton_batch_and_fuses_bias():
    torch.manual_seed(23)
    x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemv(x, weight, bias)
    expected = F.linear(x, weight, bias)
    assert actual.shape == (1, 1536)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemv
def test_bf16_gemv_small_batch_fuses_bias():
    torch.manual_seed(25)
    x = torch.randn(4, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemv(x, weight, bias)
    expected = F.linear(x, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemv
def test_bf16_gemv_uses_current_stream():
    x = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = bf16_gemv(x, weight)
        expected = F.linear(x, weight)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemv
def test_bf16_gemv_cuda_graph_replay():
    torch.manual_seed(29)
    x = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        bf16_gemv(x, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = bf16_gemv(x, weight)

    x.copy_(torch.randn_like(x))
    graph.replay()
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemv
@pytest.mark.parametrize("n,k", [(64, 7), (64, 12), (33, 100), (256, 1534)])
def test_bf16_gemv_handles_unaligned_k(n, k):
    torch.manual_seed(29)
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemv(x, weight)
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)

    x3 = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    actual3 = bf16_gemv(x3, weight)
    torch.testing.assert_close(actual3, F.linear(x3, weight), rtol=0.02, atol=0.5)


@skip_no_gemv
def test_bf16_gemv_small_batch_cuda_graph_replay():
    torch.manual_seed(31)
    x = torch.randn(8, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        bf16_gemv(x, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = bf16_gemv(x, weight)

    x.copy_(torch.randn_like(x))
    graph.replay()
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemv
@pytest.mark.parametrize(
    "make_args,error",
    [
        (
            lambda: (
                torch.randn(9, 16, device="cuda", dtype=torch.bfloat16),
                torch.randn(8, 16, device="cuda", dtype=torch.bfloat16),
            ),
            "M must",
        ),
        (
            lambda: (
                torch.randn(16, device="cuda", dtype=torch.float16),
                torch.randn(8, 16, device="cuda", dtype=torch.float16),
            ),
            "bf16",
        ),
        (
            lambda: (
                torch.randn(
                    16, device="cuda", dtype=torch.bfloat16, requires_grad=True
                ),
                torch.randn(8, 16, device="cuda", dtype=torch.bfloat16),
            ),
            "autograd",
        ),
    ],
)
def test_bf16_gemv_rejects_unsupported_inputs(make_args, error):
    with pytest.raises(RuntimeError, match=error):
        bf16_gemv(*make_args())
