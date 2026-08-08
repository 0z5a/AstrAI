"""Inference module for continuous batching.

Subpackages:
  - cache/:     KV cache (buffers, strategies, pool)
  - runtime/:   Execution + sampling (executor, CUDA graph, sampling strategies)
  - task/:      Request lifecycle + performance metrics
  - network/:   HTTP protocol handling (server, protocol, OpenAI/Anthropic builders)

Modules:
  - scheduler.py:  Continuous batching loop
  - workspace.py:  Pre-allocated GPU buffers
  - engine.py:     Facade (InferenceEngine)
"""

from astrai.inference.cache import (
    Allocator,
    KVCache,
    KVStorage,
    PagePool,
    RadixCache,
    ReqToTokenPool,
    TaskCacheManager,
    page_hash,
)
from astrai.inference.engine import InferenceEngine
from astrai.inference.network import (
    AnthropicMessage,
    BaseToolParser,
    ChatCompletionRequest,
    ChatMessage,
    FunctionDef,
    GenContext,
    MessagesRequest,
    ProtocolHandler,
    SimpleJsonToolParser,
    StopChecker,
    ToolDef,
    ToolParserFactory,
    get_app,
    run_server,
)
from astrai.inference.network.anthropic import AnthropicResponseBuilder
from astrai.inference.network.openai import OpenAIResponseBuilder
from astrai.inference.runtime.executor import Executor
from astrai.inference.runtime.sample import (
    BaseSamplingStrategy,
    FrequencyPenaltyStrategy,
    SamplingPipeline,
    TemperatureStrategy,
    TopKStrategy,
    TopPStrategy,
    sample,
)
from astrai.inference.scheduler import InferenceScheduler
from astrai.inference.task import STOP, Task, TaskManager, TaskStatus

__all__ = [
    "InferenceEngine",
    "InferenceScheduler",
    "Executor",
    "STOP",
    "Task",
    "TaskManager",
    "TaskStatus",
    "Allocator",
    "KVCache",
    "KVStorage",
    "PagePool",
    "RadixCache",
    "ReqToTokenPool",
    "TaskCacheManager",
    "page_hash",
    "sample",
    "BaseSamplingStrategy",
    "TemperatureStrategy",
    "TopKStrategy",
    "TopPStrategy",
    "FrequencyPenaltyStrategy",
    "SamplingPipeline",
    "ProtocolHandler",
    "StopChecker",
    "GenContext",
    "BaseToolParser",
    "SimpleJsonToolParser",
    "ToolParserFactory",
    "OpenAIResponseBuilder",
    "AnthropicResponseBuilder",
    "ChatMessage",
    "ChatCompletionRequest",
    "FunctionDef",
    "ToolDef",
    "AnthropicMessage",
    "MessagesRequest",
    "get_app",
    "run_server",
]
