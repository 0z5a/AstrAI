"""Benchmark fused BF16 SwiGLU against torch and unfused GEMM chains."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import click
import torch
import torch.nn.functional as F

from astrai.extension import bf16_gemm, bf16_swiglu, is_available


@dataclass(frozen=True)
class SwiGLUShape:
    name: str
    n: int
    k: int


DEFAULT_SHAPES = (
    SwiGLUShape("astrai_1b", 6912, 1536),
    SwiGLUShape("llama2_7b", 11008, 4096),
    SwiGLUShape("llama3_8b", 14336, 4096),
    SwiGLUShape("llama2_13b", 13824, 5120),
    SwiGLUShape("gpt_neox_20b", 16384, 6144),
)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise click.BadParameter("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise click.BadParameter("values must be positive integers")
    return values


def parse_shape(value: str) -> SwiGLUShape:
    parts = value.split(":")
    if len(parts) != 3 or not parts[0]:
        raise click.BadParameter("shape must use NAME:N:K")
    try:
        n, k = (int(item) for item in parts[1:])
    except ValueError as exc:
        raise click.BadParameter("N and K must be integers") from exc
    if n <= 0 or k <= 0 or k % 8:
        raise click.BadParameter("N must be positive and K positive/divisible by 8")
    return SwiGLUShape(parts[0], n, k)


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def time_operation(operation: Callable[[], torch.Tensor], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def count_cuda_kernels(
    operation: Callable[[], torch.Tensor], repeats: int = 5
) -> float:
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


def capture(operation: Callable[[], torch.Tensor]):
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()

    def replay() -> torch.Tensor:
        graph.replay()
        return output

    return replay


def make_operations(x, up_weight, gate_weight, mode: str):
    operations: dict[str, Callable[[], torch.Tensor]] = {
        "torch": lambda: F.linear(x, up_weight) * F.silu(F.linear(x, gate_weight)),
        "gemm_chain": lambda: (
            bf16_gemm(x, up_weight) * F.silu(bf16_gemm(x, gate_weight))
        ),
        "fused": lambda: bf16_swiglu(x, up_weight, gate_weight),
    }
    if mode == "graph":
        operations = {name: capture(op) for name, op in operations.items()}
    return operations


def benchmark_case(
    shape: SwiGLUShape,
    m: int,
    mode: str,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> list[dict[str, object]]:
    x = torch.randn((m, shape.k), device="cuda", dtype=torch.bfloat16) * 0.1
    scale = shape.k**-0.5
    up_weight = (
        torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16) * scale
    )
    gate_weight = (
        torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16) * scale
    )
    operations = make_operations(x, up_weight, gate_weight, mode)
    for operation in operations.values():
        for _ in range(warmup):
            operation()
    torch.cuda.synchronize()

    samples = {name: [] for name in operations}
    forward_order = tuple(operations)
    # A-B-C-C-B-A order balances cache, clock, and temperature drift.
    for _ in range(trials):
        for name in (*forward_order, *reversed(forward_order)):
            samples[name].append(time_operation(operations[name], iterations))

    with torch.no_grad():
        expected = operations["torch"]().clone()
        actual = operations["fused"]().clone()
    difference = (actual.float() - expected.float()).abs()
    max_abs_error = float(difference.max())
    mean_abs_error = float(difference.mean())
    cosine_similarity = float(
        F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    )

    results = []
    for name, operation in operations.items():
        result: dict[str, object] = {
            "shape": shape.name,
            "m": m,
            "n": shape.n,
            "k": shape.k,
            "mode": mode,
            "implementation": name,
            "cuda_kernel_launches_per_call": count_cuda_kernels(operation),
            **summarize(samples[name]),
        }
        if name == "fused":
            result.update(
                max_abs_error=max_abs_error,
                mean_abs_error=mean_abs_error,
                cosine_similarity=cosine_similarity,
            )
        results.append(result)
    return results


def device_metadata() -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": "bfloat16",
    }


def render_markdown(payload: dict[str, object]) -> str:
    metadata = payload["metadata"]
    results = payload["results"]
    assert isinstance(metadata, dict)
    assert isinstance(results, list)
    by_case = {
        (item["shape"], item["m"], item["mode"], item["implementation"]): item
        for item in results
    }
    cases = sorted({(item["shape"], item["m"], item["mode"]) for item in results})
    lines = [
        "# Fused SwiGLU benchmark",
        "",
        f"- GPU: {metadata['gpu_name']}",
        f"- Compute capability: {metadata['compute_capability']}",
        f"- PyTorch / CUDA: {metadata['torch_version']} / {metadata['cuda_version']}",
        "",
        "| Shape | M | Mode | torch ms | GEMM chain ms | fused ms | "
        "vs best unfused | fused kernels | max abs | cosine |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for shape, m, mode in cases:
        torch_item = by_case[(shape, m, mode, "torch")]
        gemm_item = by_case[(shape, m, mode, "gemm_chain")]
        fused_item = by_case[(shape, m, mode, "fused")]
        best = min(torch_item["median_ms"], gemm_item["median_ms"])
        improvement = (best / fused_item["median_ms"] - 1) * 100
        lines.append(
            f"| {shape} | {m} | {mode} | {torch_item['median_ms']:.5f} | "
            f"{gemm_item['median_ms']:.5f} | {fused_item['median_ms']:.5f} | "
            f"{improvement:+.2f}% | "
            f"{fused_item['cuda_kernel_launches_per_call']:.1f} | "
            f"{fused_item['max_abs_error']:.5f} | "
            f"{fused_item['cosine_similarity']:.8f} |"
        )
    lines.append("")
    return "\n".join(lines)


@click.command(help=__doc__)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--markdown-output", type=click.Path(path_type=Path))
@click.option("--m-values", default="1,2,4,8", show_default=True)
@click.option("--shape", "shape_values", multiple=True, help="Repeat NAME:N:K.")
@click.option("--mode", type=click.Choice(("eager", "graph", "both")), default="both")
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option(
    "--iterations", type=click.IntRange(min=1), default=100, show_default=True
)
@click.option("--trials", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def benchmark_command(
    output: Path,
    markdown_output: Path | None,
    m_values: str,
    shape_values: tuple[str, ...],
    mode: str,
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if not is_available("bf16_gemm") or not is_available("bf16_swiglu"):
        raise click.ClickException("built bf16_gemm and bf16_swiglu are required")
    shapes = tuple(parse_shape(value) for value in shape_values) or DEFAULT_SHAPES
    m_values_parsed = parse_positive_ints(m_values)
    if any(m > 8 for m in m_values_parsed):
        raise click.BadParameter("fused primitive supports M up to 8")
    modes = ("eager", "graph") if mode == "both" else (mode,)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    results = []
    with torch.inference_mode():
        for shape in shapes:
            for m in m_values_parsed:
                for current_mode in modes:
                    click.echo(
                        f"{shape.name}: M={m} N={shape.n} K={shape.k} {current_mode}"
                    )
                    results.extend(
                        benchmark_case(
                            shape,
                            m,
                            current_mode,
                            warmup=warmup,
                            iterations=iterations,
                            trials=trials,
                        )
                    )
            torch.cuda.empty_cache()

    payload: dict[str, object] = {
        "metadata": device_metadata(),
        "settings": {
            "warmup": warmup,
            "iterations": iterations,
            "trials": trials,
            "seed": seed,
            "order": "A-B-C-C-B-A",
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(payload))


if __name__ == "__main__":
    benchmark_command()
