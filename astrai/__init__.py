__version__ = "1.3.11"
__author__ = "ViperEkura"

import logging
import os

from astrai.config import (
    AutoRegressiveLMConfig,
    BaseModelConfig,
    ConfigFactory,
    EncoderConfig,
    PipelineConfig,
    TrainConfig,
)
from astrai.dataset import (
    BaseDataset,
    DatasetFactory,
    RDSampler,
    Store,
    StoreFactory,
)
from astrai.factory import BaseFactory
from astrai.inference import (
    GenerationRequest,
    InferenceEngine,
    ProtocolHandler,
    SamplingPipeline,
    get_app,
    run_server,
    sample,
)
from astrai.model import (
    AutoModel,
    AutoRegressiveLM,
    EmbeddingEncoder,
    LoRAConfig,
    inject_lora,
)
from astrai.parallel import (
    ExecutorFactory,
    get_rank,
    get_world_size,
    only_on_rank,
    spawn_parallel_fn,
)
from astrai.preprocessing import Pipeline, filter_by_length
from astrai.serialization import Checkpoint
from astrai.tokenize import AutoTokenizer, ChatTemplate
from astrai.trainer import (
    BaseScheduler,
    BaseStrategy,
    CallbackFactory,
    SchedulerFactory,
    StrategyFactory,
    TrainCallback,
    Trainer,
)


def setup_logging(level: str = "INFO"):
    """Attach a handler to the ``astrai`` logger (only, not root).

    Call once per process, e.g. at the top of CLI scripts.
    Set ``ASTR_LOG_LEVEL`` to override the default ``INFO``.
    """
    _logger = logging.getLogger("astrai")
    if _logger.handlers:
        return
    _level = getattr(
        logging, os.environ.get("ASTR_LOG_LEVEL", level).upper(), logging.INFO
    )
    _logger.setLevel(_level)
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _logger.addHandler(_handler)


__all__ = [
    "AutoRegressiveLM",
    "AutoRegressiveLMConfig",
    "AutoModel",
    "AutoTokenizer",
    "BaseDataset",
    "BaseFactory",
    "BaseModelConfig",
    "BaseScheduler",
    "BaseStrategy",
    "CallbackFactory",
    "ChatTemplate",
    "Checkpoint",
    "ConfigFactory",
    "DatasetFactory",
    "EmbeddingEncoder",
    "EncoderConfig",
    "ExecutorFactory",
    "GenerationRequest",
    "InferenceEngine",
    "LoRAConfig",
    "Pipeline",
    "PipelineConfig",
    "ProtocolHandler",
    "RDSampler",
    "SamplingPipeline",
    "SchedulerFactory",
    "Store",
    "StoreFactory",
    "StrategyFactory",
    "TrainCallback",
    "TrainConfig",
    "Trainer",
    "filter_by_length",
    "get_app",
    "get_rank",
    "get_world_size",
    "inject_lora",
    "only_on_rank",
    "run_server",
    "sample",
    "setup_logging",
    "spawn_parallel_fn",
]
