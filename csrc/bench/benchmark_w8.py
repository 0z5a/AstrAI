"""Benchmark the quantized-GEMM family (W8A16 / W8A8 / W16A16) against the
bf16 ``F.linear`` baseline.

All three modes run the NT orientation the linear path uses (bf16 / int8
activation ``[M][K]``, weight ``[N][K]``). W8A16 applies per-channel weight
scales; W8A8 additionally quantizes activations per-row (the quantize pass
is excluded from the timed GEMM — it prices the kernel, not the policy).
Agreement columns report the max error against the dequantized reference.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import torch
import torch.nn.functional as F

from astrai.extension import is_available
from astrai.extension.ops.gemm import quant_gemm
from astrai.extension.quantize import quantize_act_int8, quantize_weight_int8

# GEMM shapes as (N, K) weight mats; M comes from --m-values.
GEMM_SHAPES = (
    ("llama2_7b_qkv", 4096, 4096),
    ("llama2_7b_up_gate", 11008, 4096),
    ("llama2_7b_down", 4096, 11008),
    ("llama3_70b_up_gate", 28672, 8192),
)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise click.BadParameter("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise click.BadParameter("values must be positive integers")
    return values


def parse_shape(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) != 3 or not parts[0]:
        raise click.BadParameter("shape must use NAME:ROWS:COLS")
    try:
        rows, cols = (int(item) for item in parts[1:])
    except ValueError as exc:
        raise click.BadParameter("ROWS:COLS must be integers") from exc
    if rows <= 0 or cols <= 0:
        raise click.BadParameter("ROWS and COLS must be positive")
    return parts[0], rows, cols


def time_operation(operation: Callable[[], torch.Tensor], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


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


def benchmark_gemm(
    name: str,
    n: int,
    k: int,
    m: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, object]:
    x = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    w8, ws = quantize_weight_int8(w)
    x8, xs = quantize_act_int8(x)

    w_ref16 = w
    w_ref8 = (w8.float() * ws.unsqueeze(1)).to(torch.bfloat16)
    x_ref8 = (x8.float() * xs.unsqueeze(1)).to(torch.bfloat16)
    ref16 = F.linear(x, w_ref16).float()
    ref816 = F.linear(x, w_ref8).float()
    ref88 = F.linear(x_ref8, w_ref8).float()

    samples = measure_operations(
        {
            "bf16": lambda: F.linear(x, w),
            "w16a16": lambda: quant_gemm(x, w),
            "w8a16": lambda: quant_gemm(x, w8, b_scale=ws),
            "w8a8": lambda: quant_gemm(x8, w8, a_scale=xs, b_scale=ws),
        },
        warmup=warmup,
        iterations=iterations,
        trials=trials,
    )
    med = {key: statistics.median(vals) for key, vals in samples.items()}
    flops = 2.0 * m * n * k
    tfs = {key: flops / (ms * 1e-3) / 1e12 for key, ms in med.items()}
    err = {
        "w16a16": (quant_gemm(x, w).float() - ref16).abs().max().item(),
        "w8a16": (quant_gemm(x, w8, b_scale=ws).float() - ref816).abs().max().item(),
        "w8a8": (quant_gemm(x8, w8, a_scale=xs, b_scale=ws).float() - ref88)
        .abs()
        .max()
        .item(),
    }
    print(
        f"{name},{m}x{n}x{k},{med['bf16']:.4f},{med['w16a16']:.4f},"
        f"{med['w8a16']:.4f},{med['w8a8']:.4f},{tfs['bf16']:.1f},"
        f"{tfs['w16a16']:.1f},{tfs['w8a16']:.1f},{tfs['w8a8']:.1f},"
        f"{med['bf16'] / med['w8a16']:.2f}x,{med['bf16'] / med['w8a8']:.2f}x,"
        f"{err['w16a16']:.4f},{err['w8a16']:.4f},{err['w8a8']:.4f}"
    )
    return {
        "shape": name,
        "m": m,
        "n": n,
        "k": k,
        "median_ms": med,
        "tflops": tfs,
        "speedup_vs_bf16": {
            key: med["bf16"] / ms for key, ms in med.items() if key != "bf16"
        },
        "max_err": err,
    }


@click.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional JSON evidence path (kept out of the repository).",
)
@click.option(
    "--m-values",
    default="512,2048,4096",
    show_default=True,
    callback=lambda _c, _p, v: parse_positive_ints(v),
)
@click.option(
    "--shape",
    "shape_values",
    multiple=True,
    help="Filter defaults by name or add NAME:N:K.",
)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--trials", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def benchmark_command(
    output: Path | None,
    m_values: tuple[int, ...],
    shape_values: tuple[str, ...],
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if not is_available("gemm"):
        raise click.ClickException("the built gemm kernel is required")

    bare_names = {value for value in shape_values if ":" not in value}
    known = {shape[0] for shape in GEMM_SHAPES}
    unknown = sorted(bare_names - known)
    if unknown:
        raise click.BadParameter(f"unknown default shape names: {', '.join(unknown)}")
    specs = [parse_shape(value) for value in shape_values if ":" in value]
    if shape_values:
        by_name = {s[0]: s for s in GEMM_SHAPES if s[0] in bare_names}
        by_name.update({s[0]: s for s in specs})
        shapes = list(by_name.values())
    else:
        shapes = list(GEMM_SHAPES)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(
        "shape,mxn_xk,bf16_ms,w16a16_ms,w8a16_ms,w8a8_ms,"
        "bf16_tflops,w16a16_tflops,w8a16_tflops,w8a8_tflops,"
        "w8a16_vs_bf16,w8a8_vs_bf16,err16,err816,err88"
    )
    results = []
    with torch.inference_mode():
        for name, n, k in shapes:
            for m in m_values:
                results.append(
                    benchmark_gemm(
                        name,
                        n,
                        k,
                        m,
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
            },
            "settings": {
                "warmup": warmup,
                "iterations": iterations,
                "trials": trials,
                "seed": seed,
                "order": "A-B-C-C-B-A",
                "m_values": list(m_values),
            },
            "results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    benchmark_command()
