import pytest
import torch
import torch.nn.functional as F

from astrai.extension import bf16_gemm, is_available

GEMM_AVAILABLE = (
    torch.cuda.is_available()
    and is_available("bf16_gemm")
    and torch.cuda.get_device_capability() >= (8, 0)
)
skip_no_gemm = pytest.mark.skipif(
    not GEMM_AVAILABLE,
    reason="BF16 GEMM requires a built kernel and compute capability 8.0+",
)


def _assert_close_fp64(actual, x, weight, bias=None):
    """Compare bf16 kernel output vs fp64-exact with ulp-scaled tolerance.

    Avoids false failures from cuBLAS default bf16 split-K partial reduction
    (which can introduce ~2 ulp diffs on near-tie rounding at long K). Used
    for tiled-path tests (M > 8, K >= 4096) where the bf16 accumulation tie
    pattern may differ from cuBLAS's."""
    exact = x.double() @ weight.double().T
    if bias is not None:
        exact = exact + bias.double()
    ulp = (exact.abs() * 2**-9).clamp(min=2**-9)
    max_ulp = ((actual.double() - exact).abs() / ulp).max().item()
    assert max_ulp < 8, (
        f"max_ulp={max_ulp:.1f} exceeds 8 (exact fp32 accumulation should stay within ~2 ulps)"
    )


@skip_no_gemm
@pytest.mark.parametrize(
    "n,k",
    [(256, 1536), (1536, 1536), (6912, 1536), (1536, 6912), (100000, 1536)],
)
def test_bf16_gemm_matches_linear_shape_families(n, k):
    torch.manual_seed(17)
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemm(x, weight)
    expected = F.linear(x, weight)
    assert actual.shape == (n,)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemm
@pytest.mark.parametrize("m", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("n,k", [(256, 1536), (1536, 1536), (1536, 6912)])
def test_bf16_gemm_matches_small_decode_batches(m, n, k):
    torch.manual_seed(19 + m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemm(x, weight)
    expected = F.linear(x, weight)
    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.5)


@skip_no_gemm
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
def test_bf16_gemm_matches_common_transformer_shapes(m, n, k):
    torch.manual_seed(2026 + m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    weight.normal_(mean=0.0, std=0.02)
    actual = bf16_gemm(x, weight)
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemm
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
def test_bf16_gemm_matches_m8_edge_bands(m, n, k):
    torch.manual_seed(2026 + m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    weight.normal_(mean=0.0, std=0.02)
    actual = bf16_gemm(x, weight)
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemm
def test_bf16_gemm_preserves_singleton_batch_and_fuses_bias():
    torch.manual_seed(23)
    x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemm(x, weight, bias)
    expected = F.linear(x, weight, bias)
    assert actual.shape == (1, 1536)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemm
def test_bf16_gemm_small_batch_fuses_bias():
    torch.manual_seed(25)
    x = torch.randn(4, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemm(x, weight, bias)
    expected = F.linear(x, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemm
def test_bf16_gemm_uses_current_stream():
    x = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1536, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = bf16_gemm(x, weight)
        expected = F.linear(x, weight)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemm
def test_bf16_gemm_cuda_graph_replay():
    torch.manual_seed(29)
    x = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        bf16_gemm(x, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = bf16_gemm(x, weight)

    x.copy_(torch.randn_like(x))
    graph.replay()
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemm
@pytest.mark.parametrize("n,k", [(64, 7), (64, 12), (33, 100), (256, 1534)])
def test_bf16_gemm_handles_unaligned_k(n, k):
    torch.manual_seed(29)
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemm(x, weight)
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)

    x3 = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    actual3 = bf16_gemm(x3, weight)
    torch.testing.assert_close(actual3, F.linear(x3, weight), rtol=0.02, atol=0.5)


@skip_no_gemm
@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_bf16_gemm_accepts_complementary_misalignment(m):
    """Misaligned weight rows plus an x base chosen so the vectorized branch
    is entered with a non-16B-aligned ``x`` pointer (regression: the branch
    guard checked ``x + whead`` alignment but the uint4 view was rooted at
    ``x`` itself, faulting with a misaligned-address CUDA error)."""
    torch.manual_seed(37)
    n, k = 256, 1536
    # offset 5 elements = +10 bytes: weight rows land at 10 % 16 (whead=3)
    # and x at 10 % 16, so (x + 2*whead) % 16 == 0 selects the fast path.
    big_w = torch.randn(n * k + 8, device="cuda", dtype=torch.bfloat16)
    weight = big_w[5 : 5 + n * k].view(n, k)
    big_x = torch.randn(m * k + 8, device="cuda", dtype=torch.bfloat16)
    x = big_x[5 : 5 + m * k].view(m, k) if m > 1 else big_x[5 : 5 + k]
    assert (x.data_ptr() & 15) == 10 and (weight.data_ptr() & 15) == 10

    actual = bf16_gemm(x, weight)
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.5 if m > 1 else 0.25)


@skip_no_gemm
def test_bf16_gemm_scalar_path_handles_misaligned_weight_only():
    """Weight rows misaligned while x stays 16B-aligned take the scalar-x
    middle and must stay exact."""
    torch.manual_seed(41)
    n, k = 256, 1536
    big_w = torch.randn(n * k + 8, device="cuda", dtype=torch.bfloat16)
    weight = big_w[5 : 5 + n * k].view(n, k)
    x = torch.randn(2, k, device="cuda", dtype=torch.bfloat16)
    assert (weight.data_ptr() & 15) == 10 and (x.data_ptr() & 15) == 0

    actual = bf16_gemm(x, weight)
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=0.02, atol=0.5)


@skip_no_gemm
def test_bf16_gemm_small_batch_cuda_graph_replay():
    torch.manual_seed(31)
    x = torch.randn(8, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        bf16_gemm(x, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = bf16_gemm(x, weight)

    x.copy_(torch.randn_like(x))
    graph.replay()
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@skip_no_gemm
@pytest.mark.parametrize(
    "make_args,error",
    [
        (
            lambda: (
                torch.randn(65, 16, device="cuda", dtype=torch.bfloat16),
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
def test_bf16_gemm_rejects_unsupported_inputs(make_args, error):
    with pytest.raises(RuntimeError, match=error):
        bf16_gemm(*make_args())


# ---------------------------------------------------------------------------
# Tiled path: M in (8, 64]
# ---------------------------------------------------------------------------


@skip_no_gemm
@pytest.mark.parametrize("m", [9, 12, 16, 17, 24, 32, 33, 48, 64])
@pytest.mark.parametrize("n,k", [(256, 1536), (1536, 1536), (6912, 1536), (1536, 6912)])
def test_bf16_gemm_tiled_matches_decode_batches(m, n, k):
    torch.manual_seed(31 + m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    weight.normal_(mean=0.0, std=0.02)
    actual = bf16_gemm(x, weight)
    assert actual.shape == (m, n)
    _assert_close_fp64(actual, x, weight)


@skip_no_gemm
@pytest.mark.parametrize("m", [12, 64])
def test_bf16_gemm_tiled_matches_lm_head(m):
    # N=100000 fills the SMs with N tiles alone: the splits=1 epilogue.
    torch.manual_seed(37 + m)
    x = torch.randn(m, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(100000, 1536, device="cuda", dtype=torch.bfloat16)
    weight.normal_(mean=0.0, std=0.02)
    actual = bf16_gemm(x, weight)
    _assert_close_fp64(actual, x, weight)


@skip_no_gemm
@pytest.mark.parametrize(
    "m,n,k",
    [
        (12, 100, 72),
        (12, 96, 1536),
        (12, 1632, 1536),
        (64, 100, 72),
        (17, 200, 8),
        (33, 160, 152),
    ],
)
def test_bf16_gemm_tiled_handles_remainder_tiles(m, n, k):
    # N not a multiple of 64 (predicated epilogue columns) and K not a
    # multiple of 64 (zero-filled staging chunks).
    torch.manual_seed(41 + m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemm(x, weight)
    _assert_close_fp64(actual, x, weight)


@skip_no_gemm
@pytest.mark.parametrize("m,n,k", [(12, 1536, 1536), (64, 6912, 1536)])
def test_bf16_gemm_tiled_fuses_bias(m, n, k):
    # (12, 1536) exercises the narrow-N deep-K config; (64, 6912) the
    # wide-N default.
    torch.manual_seed(43 + m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    actual = bf16_gemm(x, weight, bias)
    _assert_close_fp64(actual, x, weight, bias)


@skip_no_gemm
def test_bf16_gemm_tiled_deterministic_across_runs():
    # Single-pass K accumulation with no atomics: reruns are bitwise
    # identical — CUDA Graph replay relies on this.
    torch.manual_seed(47)
    x = torch.randn(16, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    first = bf16_gemm(x, weight)
    second = bf16_gemm(x, weight)
    assert torch.equal(first, second)


@skip_no_gemm
def test_bf16_gemm_tiled_cuda_graph_replay():
    torch.manual_seed(53)
    x = torch.randn(16, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        bf16_gemm(x, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = bf16_gemm(x, weight)

    x.copy_(torch.randn_like(x))
    graph.replay()
    _assert_close_fp64(actual, x, weight)


@skip_no_gemm
def test_bf16_gemm_tiled_rejects_k_not_multiple_of_8():
    x = torch.randn(12, 12, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 12, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="multiple of 8"):
        bf16_gemm(x, weight)


@skip_no_gemm
def test_bf16_gemm_tiled_rejects_misaligned_x():
    # A 2-byte storage offset breaks the 16B alignment the tiled path
    # stages chunks on; the M <= 8 GEMV path still accepts it.
    k = 1536
    storage = torch.randn(12 * k + 1, device="cuda", dtype=torch.bfloat16)
    x8 = storage[1 : 1 + 8 * k].view(8, k)
    x12 = storage[1 : 1 + 12 * k].view(12, k)
    weight = torch.randn(1536, k, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        bf16_gemm(x8, weight), F.linear(x8, weight), rtol=0.02, atol=0.5
    )
    with pytest.raises(RuntimeError, match="16-byte"):
        bf16_gemm(x12, weight)
