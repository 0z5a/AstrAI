"""Frontend layer (vLLM: API process — AsyncLLM, Input/OutputProcessor).

Everything that faces the user: the engine facade, request-id minting,
tokenization, detokenization/stop handling, the output-event contract and
the engine-core client seam.  Frontend code never touches KV state or
model tensors directly.
"""

from astrai.inference.core.events import (
    FINISH_ABORTED,
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_REJECTED,
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.frontend.core_client import EngineCoreClient, InprocClient
from astrai.inference.frontend.engine import (
    GenerateResult,
    InferenceEngine,
    build_engine,
)
from astrai.inference.frontend.input_processor import InputProcessor
from astrai.inference.frontend.output_processor import (
    OutputProcessor,
    StopSequenceChecker,
)

__all__ = [
    "FINISH_ABORTED",
    "FINISH_CANCELLED",
    "FINISH_LENGTH",
    "FINISH_REJECTED",
    "FINISH_STOP_TOKEN",
    "EngineCoreClient",
    "GenerateResult",
    "InferenceEngine",
    "InprocClient",
    "InputProcessor",
    "OutputProcessor",
    "RequestError",
    "RequestFinished",
    "StopSequenceChecker",
    "TokenDelta",
    "build_engine",
]
