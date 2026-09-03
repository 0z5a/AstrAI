"""Benchmark decode-time linear shapes before enabling custom GEMV dispatch.

The benchmark deliberately calls ``torch.nn.functional.linear`` directly. It
establishes the per-architecture cuBLAS baseline that later GEMV primitives and
dispatch decisions must beat.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import click
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LinearShape:
    name: str
    n: int
    k: int


DEFAULT_SHAPES = (
    LinearShape("q_proj", 1536, 1536),
    LinearShape("k_proj", 256, 1536),
    LinearShape("v_proj", 256, 1536),
    LinearShape("attn_out", 1536, 1536),
    LinearShape("mlp_up", 6912, 1536),
    LinearShape("mlp_gate", 6912, 1536),
    LinearShape("mlp_down", 1536, 6912),
    LinearShape("lm_head", 100000, 1536),
)
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_positive_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, duplicate-free list of positive integers."""
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise click.BadParameter("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise click.BadParameter("values must be positive integers")
    return values


def parse_shape(value: str) -> LinearShape:
    """Parse NAME:N:K into a benchmark shape."""
    parts = value.split(":")
    if len(parts) != 3 or not parts[0]:
        raise click.BadParameter("shape must use NAME:N:K")
    try:
        n, k = (int(item) for item in parts[1:])
    except ValueError as exc:
        raise click.BadParameter("N and K must be integers") from exc
    if n <= 0 or k <= 0:
        raise click.BadParameter("N and K must be positive")
    return LinearShape(parts[0], n, k)


def estimate_io_bytes(
    m: int, n: int, k: int, element_size: int, *, has_bias: bool
) -> int:
    """Estimate bytes touched once by Y[M,N] = X[M,K] @ W[N,K].T."""
    elements = m * k + n * k + m * n
    if has_bias:
        elements += n
    return elements * element_size


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latency(samples_ms: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples_ms),
        "p90_ms": percentile(samples_ms, 0.90),
        "p99_ms": percentile(samples_ms, 0.99),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def measure_cuda_ms(
    operation: Callable[[], torch.Tensor], *, warmup: int, iterations: int, trials: int
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def count_cuda_kernels(
    operation: Callable[[], torch.Tensor], repeats: int = 5
) -> float:
    """Profile a few calls and return the average device events per call."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        acc_events=True,
    ) as profile:
        for _ in range(repeats):
            operation()
        torch.cuda.synchronize()

    device_type = torch.autograd.DeviceType.CUDA
    events = [event for event in profile.events() if event.device_type == device_type]
    return len(events) / repeats


def capture_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        F.linear(x, weight, bias)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = F.linear(x, weight, bias)
    return graph, output


def benchmark_case(
    shape: LinearShape,
    m: int,
    *,
    dtype: torch.dtype,
    mode: str,
    bias_enabled: bool,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, object]:
    x = torch.randn((m, shape.k), device="cuda", dtype=dtype)
    weight = torch.randn((shape.n, shape.k), device="cuda", dtype=dtype)
    bias = torch.randn(shape.n, device="cuda", dtype=dtype) if bias_enabled else None

    graph = None
    graph_output = None
    if mode == "graph":
        graph, graph_output = capture_linear(x, weight, bias)

        def operation() -> torch.Tensor:
            graph.replay()
            return graph_output

    else:

        def operation() -> torch.Tensor:
            return F.linear(x, weight, bias)

    samples_ms = measure_cuda_ms(
        operation, warmup=warmup, iterations=iterations, trials=trials
    )
    latency = summarize_latency(samples_ms)
    io_bytes = estimate_io_bytes(
        m, shape.n, shape.k, x.element_size(), has_bias=bias is not None
    )
    median_seconds = latency["median_ms"] / 1000

    result: dict[str, object] = {
        "name": shape.name,
        "m": m,
        "n": shape.n,
        "k": shape.k,
        "mode": mode,
        "bias": bias is not None,
        "estimated_io_bytes": io_bytes,
        "effective_bandwidth_gbps": io_bytes / median_seconds / 1e9,
        "cuda_kernel_launches_per_call": count_cuda_kernels(operation),
        **latency,
        "samples_ms": samples_ms,
    }
    return result


def render_markdown(payload: dict[str, object]) -> str:
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    results = payload["results"]
    assert isinstance(results, list)

    lines = [
        "# Decode linear baseline",
        "",
        f"- GPU: {metadata['gpu_name']}",
        f"- Compute capability: {metadata['compute_capability']}",
        f"- PyTorch / CUDA: {metadata['torch_version']} / {metadata['cuda_version']}",
        f"- Dtype: {metadata['dtype']}",
        "",
        "| Layer | M | N | K | Mode | Median (ms) | p99 (ms) | GB/s | CUDA kernels/call |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in results:
        assert isinstance(item, dict)
        lines.append(
            "| {name} | {m} | {n} | {k} | {mode} | {median_ms:.4f} | "
            "{p99_ms:.4f} | {effective_bandwidth_gbps:.1f} | "
            "{cuda_kernel_launches_per_call:.2f} |".format(**item)
        )
    lines.append("")
    return "\n".join(lines)


def device_metadata(dtype_name: str) -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": dtype_name,
    }


@click.command(help=__doc__)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--markdown-output", type=click.Path(path_type=Path))
@click.option("--m-values", default="1,2,4,8,16,32", show_default=True)
@click.option(
    "--shape",
    "shape_values",
    multiple=True,
    help="Override defaults with repeatable NAME:N:K shapes.",
)
@click.option("--dtype", type=click.Choice(tuple(DTYPES)), default="bfloat16")
@click.option("--mode", type=click.Choice(("eager", "graph", "both")), default="both")
@click.option("--bias/--no-bias", default=False)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option(
    "--iterations", type=click.IntRange(min=1), default=100, show_default=True
)
@click.option("--trials", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def benchmark_command(
    output: Path,
    markdown_output: Path | None,
    m_values: str,
    shape_values: tuple[str, ...],
    dtype: str,
    mode: str,
    bias: bool,
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")

    parsed_m = parse_positive_ints(m_values)
    shapes = tuple(parse_shape(item) for item in shape_values) or DEFAULT_SHAPES
    modes = ("eager", "graph") if mode == "both" else (mode,)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    results = []
    for shape in shapes:
        for m in parsed_m:
            for current_mode in modes:
                click.echo(
                    f"{shape.name}: M={m} N={shape.n} K={shape.k} {current_mode}"
                )
                results.append(
                    benchmark_case(
                        shape,
                        m,
                        dtype=DTYPES[dtype],
                        mode=current_mode,
                        bias_enabled=bias,
                        warmup=warmup,
                        iterations=iterations,
                        trials=trials,
                    )
                )

    payload: dict[str, object] = {
        "schema_version": 1,
        "metadata": device_metadata(dtype),
        "parameters": {
            "m_values": list(parsed_m),
            "shapes": [asdict(shape) for shape in shapes],
            "modes": list(modes),
            "bias": bias,
            "warmup": warmup,
            "iterations": iterations,
            "trials": trials,
            "seed": seed,
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    benchmark_command()
