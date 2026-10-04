"""Shared CUDA timing and interleaved sampling for benchmark scripts."""

from __future__ import annotations

import math
import statistics
from typing import Callable, Hashable, TypeVar

import torch

_Key = TypeVar("_Key", bound=Hashable)


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
    operations: dict[_Key, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[_Key, list[float]]:
    for operation in operations.values():
        for _ in range(warmup):
            operation()
    torch.cuda.synchronize()

    samples: dict[_Key, list[float]] = {name: [] for name in operations}
    order = tuple(operations)
    # A-B-B-A (or A-B-C-C-B-A) balances cache, clock, and temperature drift.
    for _ in range(trials):
        for name in (*order, *reversed(order)):
            samples[name].append(time_operation(operations[name], iterations))
    return samples
