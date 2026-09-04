"""Benchmark the BF16 GEMM primitive and guarded linear dispatcher.

The kernel suite covers AstrAI's native projections plus common LLaMA and
GPT-NeoX matrix shapes. The chain suite is a synthetic projection/MLP chain;
it measures dispatcher overhead and dependent MLP work, but is deliberately
not presented as a whole-model throughput benchmark.
"""

import argparse
import gc
import json
import math
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from astrai.extension import bf16_gemm, is_available, linear


@dataclass(frozen=True)
class Shape:
    label: str
    n: int
    k: int


@dataclass(frozen=True)
class Chain:
    label: str
    hidden: int
    kv: int
    intermediate: int
    fused_qkv: bool = False
    gated_mlp: bool = True


@dataclass(frozen=True)
class Timing:
    median_ms: float
    p90_ms: float


ASTRAI_SHAPES = (
    Shape("astrai_qkv", 256, 1536),
    Shape("astrai_square", 1536, 1536),
    Shape("astrai_up_gate", 6912, 1536),
    Shape("astrai_down", 1536, 6912),
    Shape("astrai_lm_head", 100000, 1536),
)

TRADITIONAL_SHAPES = (
    Shape("llama2_7b_qo", 4096, 4096),
    Shape("llama2_7b_up_gate", 11008, 4096),
    Shape("llama2_7b_down", 4096, 11008),
    Shape("llama3_8b_kv", 1024, 4096),
    Shape("llama3_8b_up_gate", 14336, 4096),
    Shape("llama3_8b_down", 4096, 14336),
    Shape("llama2_13b_qo", 5120, 5120),
    Shape("llama2_13b_up_gate", 13824, 5120),
    Shape("llama2_13b_down", 5120, 13824),
    Shape("gpt_neox_up", 16384, 4096),
    Shape("gpt_neox_down", 4096, 16384),
    Shape("qwen2_7b_kv", 512, 3584),
    Shape("qwen2_7b_qo", 3584, 3584),
    Shape("qwen2_7b_up_gate", 18944, 3584),
    Shape("qwen2_7b_down", 3584, 18944),
    Shape("llama3_70b_kv", 1024, 8192),
    Shape("llama3_70b_qo", 8192, 8192),
    Shape("llama3_70b_up_gate", 28672, 8192),
    Shape("llama3_70b_down", 8192, 28672),
    Shape("opt_1_3b_qkvo", 2048, 2048),
    Shape("opt_1_3b_up", 8192, 2048),
    Shape("opt_1_3b_down", 2048, 8192),
)

CHAINS = (
    Chain("llama2_7b", 4096, 4096, 11008),
    Chain("llama3_8b", 4096, 1024, 14336),
    Chain("llama2_13b", 5120, 5120, 13824),
    Chain("gpt_neox_20b", 4096, 4096, 16384, fused_qkv=True),
    Chain("qwen2_7b", 3584, 512, 18944),
    Chain("llama3_70b", 8192, 1024, 28672),
    Chain("opt_1_3b", 2048, 2048, 8192, gated_mlp=False),
)


def _elapsed_ms(fn: Callable[[], torch.Tensor], inner: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / inner


def _timing(values: list[float]) -> Timing:
    ordered = sorted(values)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return Timing(statistics.median(ordered), ordered[p90_index])


def _measure_pair(
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    *,
    warmup: int,
    samples: int,
    inner: int,
    prepare_baseline: Callable[[], None] = lambda: None,
    prepare_candidate: Callable[[], None] = lambda: None,
) -> tuple[Timing, Timing]:
    cases = (
        ("baseline", prepare_baseline, baseline),
        ("candidate", prepare_candidate, candidate),
    )
    for iteration in range(warmup):
        _, prepare, fn = cases[iteration % 2]
        prepare()
        fn()
    torch.cuda.synchronize()

    values: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for sample in range(samples):
        order = cases if sample % 2 == 0 else tuple(reversed(cases))
        for label, prepare, fn in order:
            prepare()
            values[label].append(_elapsed_ms(fn, inner))
    return _timing(values["baseline"]), _timing(values["candidate"])


def _print_header() -> None:
    print(
        "suite,label,m,n,k,torch_median_ms,torch_p90_ms,"
        "candidate_median_ms,candidate_p90_ms,speedup_pct,"
        "max_abs,relative_l2,argmax_equal"
    )


def _print_result(
    suite: str,
    label: str,
    m: int,
    n: int,
    k: int,
    baseline: Timing,
    candidate: Timing,
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, object]:
    difference = actual.float() - reference.float()
    max_abs = difference.abs().max().item()
    relative_l2 = difference.norm().item() / max(reference.float().norm().item(), 1e-12)
    argmax_equal = torch.equal(actual.argmax(dim=-1), reference.argmax(dim=-1))
    speedup = (baseline.median_ms / candidate.median_ms - 1.0) * 100.0
    result: dict[str, object] = {
        "suite": suite,
        "label": label,
        "m": m,
        "n": n,
        "k": k,
        "torch_median_ms": baseline.median_ms,
        "torch_p90_ms": baseline.p90_ms,
        "candidate_median_ms": candidate.median_ms,
        "candidate_p90_ms": candidate.p90_ms,
        "speedup_pct": speedup,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "argmax_equal": argmax_equal,
    }
    print(
        f"{suite},{label},{m},{n},{k},"
        f"{baseline.median_ms:.6f},{baseline.p90_ms:.6f},"
        f"{candidate.median_ms:.6f},{candidate.p90_ms:.6f},"
        f"{speedup:+.2f},{max_abs:.6f},{relative_l2:.8f},"
        f"{str(argmax_equal).lower()}",
        flush=True,
    )
    return result


def _weight(n: int, k: int, device: torch.device, std: float) -> torch.Tensor:
    weight = torch.empty((n, k), device=device, dtype=torch.bfloat16)
    weight.normal_(mean=0.0, std=std)
    return weight.requires_grad_(True)


def _kernel_functions(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    def baseline() -> torch.Tensor:
        return F.linear(x, weight)

    def candidate() -> torch.Tensor:
        return bf16_gemm(x, weight.detach())

    return baseline, candidate


def benchmark_kernels(
    args: argparse.Namespace, device: torch.device
) -> list[dict[str, object]]:
    if args.family == "astrai":
        shapes = ASTRAI_SHAPES
    elif args.family == "traditional":
        shapes = TRADITIONAL_SHAPES
    else:
        shapes = ASTRAI_SHAPES + TRADITIONAL_SHAPES
    if args.shape_label:
        requested = set(args.shape_label)
        shapes = tuple(shape for shape in shapes if shape.label in requested)
        missing = requested - {shape.label for shape in shapes}
        if missing:
            raise ValueError(f"unknown shape labels: {', '.join(sorted(missing))}")

    results: list[dict[str, object]] = []
    for shape in shapes:
        weight = _weight(shape.n, shape.k, device, args.weight_std)
        for m in args.m:
            x = torch.randn((m, shape.k), device=device, dtype=torch.bfloat16)
            baseline_fn, candidate_fn = _kernel_functions(x, weight)
            with torch.inference_mode():
                reference = baseline_fn()
                actual = candidate_fn()
                baseline, candidate = _measure_pair(
                    baseline_fn,
                    candidate_fn,
                    warmup=args.warmup,
                    samples=args.samples,
                    inner=args.inner,
                )
            results.append(
                _print_result(
                    "kernel",
                    shape.label,
                    m,
                    shape.n,
                    shape.k,
                    baseline,
                    candidate,
                    reference,
                    actual,
                )
            )
            del baseline_fn, candidate_fn, x, reference, actual
        del weight
        gc.collect()
        torch.cuda.empty_cache()
    return results


def _set_mode(mode: str) -> None:
    os.environ["ASTRAI_GEMM"] = mode


def _chain_weights(
    spec: Chain, device: torch.device, std: float
) -> dict[str, torch.Tensor]:
    weights = {
        "o": _weight(spec.hidden, spec.hidden, device, std),
        "up": _weight(spec.intermediate, spec.hidden, device, std),
        "down": _weight(spec.hidden, spec.intermediate, device, std),
    }
    if spec.fused_qkv:
        weights["qkv"] = _weight(3 * spec.hidden, spec.hidden, device, std)
    else:
        weights.update(
            {
                "q": _weight(spec.hidden, spec.hidden, device, std),
                "k": _weight(spec.kv, spec.hidden, device, std),
                "v": _weight(spec.kv, spec.hidden, device, std),
            }
        )
        if spec.gated_mlp:
            weights["gate"] = _weight(spec.intermediate, spec.hidden, device, std)
    return weights


def _chain_fn(
    x: torch.Tensor, weights: dict[str, torch.Tensor], spec: Chain
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        output_projection = linear(x, weights["o"])
        up = linear(x, weights["up"])
        if spec.fused_qkv:
            attention_projection = linear(x, weights["qkv"])[..., : x.shape[-1]]
            hidden = F.gelu(up)
        else:
            attention_projection = linear(x, weights["q"])
            linear(x, weights["k"])
            linear(x, weights["v"])
            if spec.gated_mlp:
                gate = linear(x, weights["gate"])
                hidden = F.silu(gate) * up
            else:
                hidden = F.gelu(up)
        down = linear(hidden, weights["down"])
        return attention_projection + output_projection + down

    return run


def benchmark_chains(
    args: argparse.Namespace, device: torch.device
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    chains = CHAINS
    if args.chain_label:
        requested = set(args.chain_label)
        chains = tuple(chain for chain in chains if chain.label in requested)
        missing = requested - {chain.label for chain in chains}
        if missing:
            raise ValueError(f"unknown chain labels: {', '.join(sorted(missing))}")
    for spec in chains:
        weights = _chain_weights(spec, device, args.weight_std)
        for m in args.m:
            x = torch.randn((m, spec.hidden), device=device, dtype=torch.bfloat16)
            run = _chain_fn(x, weights, spec)
            with torch.inference_mode():
                _set_mode("0")
                reference = run()
                _set_mode(args.candidate_mode)
                actual = run()
                baseline, candidate = _measure_pair(
                    run,
                    run,
                    warmup=args.warmup,
                    samples=args.samples,
                    inner=args.chain_inner,
                    prepare_baseline=lambda: _set_mode("0"),
                    prepare_candidate=lambda: _set_mode(args.candidate_mode),
                )
            results.append(
                _print_result(
                    "synthetic_chain",
                    spec.label,
                    m,
                    spec.hidden,
                    spec.intermediate,
                    baseline,
                    candidate,
                    reference,
                    actual,
                )
            )
            del x, reference, actual
        del weights
        gc.collect()
        torch.cuda.empty_cache()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("kernel", "chain", "all"), default="all")
    parser.add_argument(
        "--family", choices=("astrai", "traditional", "all"), default="all"
    )
    parser.add_argument(
        "--m", type=int, nargs="+", choices=(1, 2, 4, 8), default=(1, 2, 4, 8)
    )
    parser.add_argument(
        "--shape-label",
        action="append",
        help="limit the kernel suite to one or more named shape labels",
    )
    parser.add_argument(
        "--chain-label",
        action="append",
        help="limit the chain suite to one or more named model families",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--inner", type=int, default=100)
    parser.add_argument("--chain-inner", type=int, default=20)
    parser.add_argument(
        "--candidate-mode",
        choices=("auto", "1"),
        default="auto",
        help="dispatcher mode for the candidate side of the chain suite",
    )
    parser.add_argument("--weight-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output; stdout always retains the compact CSV table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not is_available("bf16_gemm"):
        raise RuntimeError("benchmark requires CUDA and the built bf16_gemm extension")
    if args.warmup < 0 or args.samples < 1 or args.inner < 1 or args.chain_inner < 1:
        raise ValueError("warmup must be non-negative and sample/inner counts positive")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    properties = torch.cuda.get_device_properties(device)
    print(
        f"# device={properties.name}, capability={properties.major}.{properties.minor}, "
        f"seed={args.seed}, weight_std={args.weight_std}"
    )
    _print_header()
    results: list[dict[str, object]] = []
    if args.suite in ("kernel", "all"):
        results.extend(benchmark_kernels(args, device))
    if args.suite in ("chain", "all"):
        results.extend(benchmark_chains(args, device))
    if args.output is not None:
        payload = {
            "environment": {
                "device": properties.name,
                "capability": f"{properties.major}.{properties.minor}",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "parameters": {
                "suite": args.suite,
                "family": args.family,
                "m": args.m,
                "shape_labels": args.shape_label,
                "chain_labels": args.chain_label,
                "candidate_mode": args.candidate_mode,
                "seed": args.seed,
                "weight_std": args.weight_std,
                "warmup": args.warmup,
                "samples": args.samples,
                "inner": args.inner,
                "chain_inner": args.chain_inner,
            },
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
