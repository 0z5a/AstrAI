"""Execution primitives: forward passes, CUDA graphs, and sampling."""

from astrai.inference.runtime.executor import Executor
from astrai.inference.runtime.graph import CudaGraphContext
from astrai.inference.runtime.sample import (
    BaseSamplingStrategy,
    FrequencyPenaltyStrategy,
    SamplingPipeline,
    TemperatureStrategy,
    TopKStrategy,
    TopPStrategy,
    sample,
)

__all__ = [
    "Executor",
    "CudaGraphContext",
    "BaseSamplingStrategy",
    "FrequencyPenaltyStrategy",
    "SamplingPipeline",
    "TemperatureStrategy",
    "TopKStrategy",
    "TopPStrategy",
    "sample",
]
