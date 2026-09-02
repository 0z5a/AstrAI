import click
import pytest

from scripts.tools.benchmark_gemv import (
    LinearShape,
    estimate_io_bytes,
    parse_positive_ints,
    parse_shape,
    percentile,
    render_markdown,
)


def test_parse_positive_ints_preserves_order_and_deduplicates():
    assert parse_positive_ints("1, 4,1,8") == (1, 4, 8)


@pytest.mark.parametrize("value", ["", "0", "1,-2", "one"])
def test_parse_positive_ints_rejects_invalid_values(value):
    with pytest.raises(click.BadParameter):
        parse_positive_ints(value)


def test_parse_shape():
    assert parse_shape("mlp_down:1536:6912") == LinearShape("mlp_down", 1536, 6912)


@pytest.mark.parametrize("value", ["", "q:1", "q:x:2", "q:0:2"])
def test_parse_shape_rejects_invalid_values(value):
    with pytest.raises(click.BadParameter):
        parse_shape(value)


def test_estimate_io_bytes_accounts_for_bias():
    without_bias = estimate_io_bytes(2, 3, 4, 2, has_bias=False)
    with_bias = estimate_io_bytes(2, 3, 4, 2, has_bias=True)
    assert without_bias == (8 + 12 + 6) * 2
    assert with_bias == without_bias + 3 * 2


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0], 0.9) == pytest.approx(1.9)


def test_render_markdown_uses_json_payload_values():
    payload = {
        "metadata": {
            "gpu_name": "NVIDIA L20",
            "compute_capability": "8.9",
            "torch_version": "2.x",
            "cuda_version": "12.x",
            "dtype": "bfloat16",
        },
        "results": [
            {
                "name": "q_proj",
                "m": 1,
                "n": 1536,
                "k": 1536,
                "mode": "eager",
                "median_ms": 0.1,
                "p99_ms": 0.2,
                "effective_bandwidth_gbps": 47.0,
                "cuda_kernel_launches_per_call": 1.0,
            }
        ],
    }
    markdown = render_markdown(payload)
    assert "NVIDIA L20" in markdown
    assert "| q_proj | 1 | 1536 | 1536 | eager | 0.1000 |" in markdown
