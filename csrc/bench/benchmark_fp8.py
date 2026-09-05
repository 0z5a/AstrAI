"""Benchmark the FP8 quantize and GEMM kernels against torch baselines.

Suites (--suite): quantize (plain / delayed-scaling ring / dual-orientation
entries vs the aten float8 cast) and gemm (pre-quantized ``mm_fp8`` in the
NT orientation the fp8 linear path uses, vs bf16 ``F.linear``). GEMM
agreement reports both kernel error (vs the fp32 dequantized fp8 product)
and format error (that product vs the bf16 matmul). FP8 MMA requires
compute capability 89+.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import torch
import torch.nn.functional as F

from astrai.extension import is_available
from astrai.extension.ops.fp8 import mm_fp8, quantize, quantize_dual

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}


@dataclass(frozen=True)
class MatrixShape:
    name: str
    rows: int
    cols: int


QUANTIZE_SHAPES = (
    MatrixShape("astrai_1b_act", 2048, 1536),
    MatrixShape("llama2_7b_act", 2048, 4096),
    MatrixShape("llama2_7b_down_w", 4096, 11008),
    MatrixShape("llama3_70b_act", 2048, 8192),
)

# GEMM shapes as (N, K) weight mats; M comes from --m-values.
GEMM_SHAPES = (
    MatrixShape("llama2_7b_qkv", 4096, 4096),
    MatrixShape("llama2_7b_up_gate", 11008, 4096),
    MatrixShape("llama2_7b_down", 4096, 11008),
    MatrixShape("llama3_70b_up_gate", 28672, 8192),
)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise click.BadParameter("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise click.BadParameter("values must be positive integers")
    return values


def parse_shape(value: str) -> MatrixShape:
    parts = value.split(":")
    if len(parts) != 3 or not parts[0]:
        raise click.BadParameter("shape must use NAME:ROWS:COLS")
    try:
        rows, cols = (int(item) for item in parts[1:])
    except ValueError as exc:
        raise click.BadParameter("ROWS and COLS must be integers") from exc
    if rows <= 0 or cols <= 0:
        raise click.BadParameter("ROWS and COLS must be positive")
    return MatrixShape(parts[0], rows, cols)


def time_operation(operation: Callable[[], torch.Tensor], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
    }


def measure_operations(
    operations: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, list[float]]:
    for operation in operations.values():
        for _ in range(warmup):
            operation()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in operations}
    order = tuple(operations)
    # A-B-C-C-B-A order balances cache, clock, and temperature drift.
    for _ in range(trials):
        for name in (*order, *reversed(order)):
            samples[name].append(time_operation(operations[name], iterations))
    return samples


def quant_step(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Quantization step (dequant scale) from the current amax; ``quantize``
    takes its reciprocal as the multiplier."""
    amax = x.abs().amax().to(torch.float32).clamp_min(1e-12)
    return amax / FP8_MAX[fmt]


def benchmark_quantize(
    shape: MatrixShape,
    fmt: str,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, object]:
    x = torch.randn(shape.rows, shape.cols, device="cuda", dtype=torch.bfloat16) * 0.1
    multiplier = quant_step(x, fmt).reciprocal()
    fp8_dtype = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    ring_state = torch.zeros(16 + 4, dtype=torch.float32, device="cuda")

    operations: dict[str, Callable[[], torch.Tensor]] = {
        "torch_cast": lambda: x.to(fp8_dtype),
        "plain": lambda: quantize(x, multiplier, fmt)[0],
        "ring": lambda: quantize(
            x,
            multiplier,
            fmt,
            ring_state=ring_state,
            hist_idx=0,
            fp8_max=FP8_MAX[fmt],
        )[0],
        "dual": lambda: quantize_dual(x, multiplier, fmt)[0],
    }
    samples = measure_operations(
        operations, warmup=warmup, iterations=iterations, trials=trials
    )

    with torch.no_grad():
        step = quant_step(x, fmt)
        x8, _ = quantize(x, multiplier, fmt)
        dequant_error = float((x8.to(torch.float32) * step - x.float()).abs().max())
        dual_t = quantize_dual(x, multiplier, fmt)[1]
        dual_matches = bool(torch.equal(dual_t.t().contiguous(), x8.contiguous()))

    io_bytes = 3 * x.numel() + 4  # bf16 read + fp8 write + f32 amax
    result: dict[str, object] = {
        "suite": "quantize",
        "shape": shape.name,
        "rows": shape.rows,
        "cols": shape.cols,
        "fmt": fmt,
        "estimated_io_bytes": io_bytes,
        "dequant_max_abs_error": dequant_error,
        "dual_transpose_matches": dual_matches,
    }
    for name, samples_ms in samples.items():
        latency = summarize(samples_ms)
        bytes_per_call = io_bytes + x.numel() if name == "dual" else io_bytes
        result[name] = {
            "effective_bandwidth_gbps": bytes_per_call
            / (latency["median_ms"] / 1000)
            / 1e9,
            **latency,
        }
    speedup = (
        result["torch_cast"]["median_ms"] / result["plain"]["median_ms"] - 1.0
    ) * 100.0
    print(
        f"quantize,{shape.name},{shape.rows}x{shape.cols},"
        f"{result['torch_cast']['median_ms']:.4f},{result['plain']['median_ms']:.4f},"
        f"{result['ring']['median_ms']:.4f},{result['dual']['median_ms']:.4f},"
        f"{speedup:+.1f}%,{dequant_error:.5f},{dual_matches}"
    )
    return result


def benchmark_gemm(
    shape: MatrixShape,
    m: int,
    fmt: str,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, object]:
    x = torch.randn(m, shape.cols, device="cuda", dtype=torch.bfloat16) * 0.1
    w = (
        torch.randn(shape.rows, shape.cols, device="cuda", dtype=torch.bfloat16)
        * shape.cols**-0.5
    )
    sx, sw = quant_step(x, fmt), quant_step(w, fmt)
    with torch.no_grad():
        x8, _ = quantize(x, sx.reciprocal(), fmt)
        w8, _ = quantize(w, sw.reciprocal(), fmt)
    dequant_scale = sx * sw

    def torch_op() -> torch.Tensor:
        return F.linear(x, w)

    def fp8_op() -> torch.Tensor:
        return mm_fp8(x8, w8, dequant_scale, trans_b=True)

    operations = {"torch_bf16": torch_op, "fp8": fp8_op}
    samples = measure_operations(
        operations, warmup=warmup, iterations=iterations, trials=trials
    )

    with torch.no_grad():
        actual = fp8_op().float()
        dequant_reference = (
            x8.to(torch.float32) @ w8.to(torch.float32).t() * dequant_scale
        )
        bf16_reference = torch_op().float()
        kernel_difference = actual - dequant_reference
        format_difference = dequant_reference - bf16_reference
    io_bytes = m * shape.cols + shape.rows * shape.cols + 2 * m * shape.rows

    result: dict[str, object] = {
        "suite": "gemm",
        "shape": shape.name,
        "m": m,
        "n": shape.rows,
        "k": shape.cols,
        "fmt": fmt,
        "estimated_io_bytes": io_bytes,
        "kernel_max_abs": float(kernel_difference.abs().max()),
        "kernel_rel_l2": float(
            kernel_difference.norm() / dequant_reference.norm().clamp_min(1e-12)
        ),
        "format_rel_l2": float(
            format_difference.norm() / bf16_reference.norm().clamp_min(1e-12)
        ),
    }
    for name, samples_ms in samples.items():
        latency = summarize(samples_ms)
        result[name] = {
            "effective_bandwidth_gbps": io_bytes / (latency["median_ms"] / 1000) / 1e9,
            **latency,
        }
    speedup = (
        result["torch_bf16"]["median_ms"] / result["fp8"]["median_ms"] - 1.0
    ) * 100.0
    print(
        f"gemm,{shape.name},{m}x{shape.rows}x{shape.cols},"
        f"{result['torch_bf16']['median_ms']:.4f},{result['fp8']['median_ms']:.4f},"
        f"{speedup:+.1f}%,{result['kernel_max_abs']:.4f},"
        f"{result['kernel_rel_l2']:.6f},{result['format_rel_l2']:.6f}"
    )
    return result


@click.command(help=__doc__)
@click.option("--output", type=click.Path(path_type=Path), help="Optional JSON output.")
@click.option(
    "--suite",
    "suites",
    type=click.Choice(("quantize", "gemm", "all")),
    multiple=True,
    default=("all",),
    show_default=True,
)
@click.option("--fmt", type=click.Choice(("e4m3", "e5m2")), default="e4m3")
@click.option("--m-values", default="512,2048,4096", show_default=True)
@click.option(
    "--shape",
    "shape_values",
    multiple=True,
    help="Filter defaults by bare name (either suite), or add/override with "
    "NAME:ROWS:COLS.",
)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option(
    "--iterations", type=click.IntRange(min=1), default=100, show_default=True
)
@click.option("--trials", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def benchmark_command(
    output: Path | None,
    suites: tuple[str, ...],
    fmt: str,
    m_values: str,
    shape_values: tuple[str, ...],
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if not is_available("quantize"):
        raise click.ClickException(
            "the built quantize kernel is required (compute capability 89+)"
        )
    selected = ("quantize", "gemm") if "all" in suites else tuple(dict.fromkeys(suites))
    m_values_parsed = parse_positive_ints(m_values)

    # Any --shape selection replaces the defaults for both suites: a bare
    # name keeps that suite's matching default, a NAME:ROWS:COLS spec
    # overrides the same-name default or adds a new one.
    bare_names = {value for value in shape_values if ":" not in value}
    known = {shape.name for shape in QUANTIZE_SHAPES + GEMM_SHAPES}
    unknown = sorted(bare_names - known)
    if unknown:
        raise click.BadParameter(f"unknown default shape names: {', '.join(unknown)}")
    specs = [parse_shape(value) for value in shape_values if ":" in value]

    def resolve_shapes(defaults: tuple[MatrixShape, ...]) -> list[MatrixShape]:
        if not shape_values:
            return list(defaults)
        by_name = {shape.name: shape for shape in defaults if shape.name in bare_names}
        for spec in specs:
            by_name[spec.name] = spec
        return list(by_name.values())

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    results = []
    with torch.inference_mode():
        if "quantize" in selected:
            print(
                "suite,shape,size,cast_ms,plain_ms,ring_ms,dual_ms,vs_cast,"
                "dequant_max,dual_ok"
            )
            for shape in resolve_shapes(QUANTIZE_SHAPES):
                results.append(
                    benchmark_quantize(
                        shape, fmt, warmup=warmup, iterations=iterations, trials=trials
                    )
                )
                torch.cuda.empty_cache()
        if "gemm" in selected:
            print(
                "suite,shape,mxn_xk,bf16_ms,fp8_ms,speedup,kernel_max,"
                "kernel_rel_l2,format_rel_l2"
            )
            for shape in resolve_shapes(GEMM_SHAPES):
                for m in m_values_parsed:
                    results.append(
                        benchmark_gemm(
                            shape,
                            m,
                            fmt,
                            warmup=warmup,
                            iterations=iterations,
                            trials=trials,
                        )
                    )
                torch.cuda.empty_cache()

    if output is not None:
        props = torch.cuda.get_device_properties(0)
        payload = {
            "metadata": {
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "fmt": fmt,
            },
            "settings": {
                "warmup": warmup,
                "iterations": iterations,
                "trials": trials,
                "seed": seed,
                "order": "A-B-C-C-B-A",
                "suites": list(selected),
                "m_values": list(m_values_parsed),
            },
            "results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    benchmark_command()
