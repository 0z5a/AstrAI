"""Execution primitives: forward passes, CUDA graphs, and sampling."""

from astrai.inference.runtime.graph import CudaGraphContext
from astrai.inference.runtime.model_runner import GPUModelRunner
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
    "GPUModelRunner",
    "CudaGraphContext",
    "BaseSamplingStrategy",
    "FrequencyPenaltyStrategy",
    "SamplingPipeline",
    "TemperatureStrategy",
    "TopKStrategy",
    "TopPStrategy",
    "sample",
]
