"""Shared fixtures for inference tests."""

from typing import Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from astrai.inference import get_app
from astrai.inference.core.scheduler import Scheduler
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.helpers import FakeTokenizer, make_rollout_config


def make_cpu_scheduler(
    model,
    tokenizer=None,
    *,
    max_batch_size: int = 8,
    max_seq_len: int = 64,
    enable_overlap: bool = False,
    enable_cuda_graph: bool = False,
    device: Optional[str] = None,
    **overrides,
) -> Scheduler:
    """Standard test scheduler: CPU-safe defaults, arbitrary overrides.

    Every keyword lands in ``Scheduler(...)`` verbatim, so tests keep direct
    control of page_size/kv_tokens/token_budget while sharing the no-graph,
    torch-native baseline that keeps the suite CPU-runnable.
    """
    if tokenizer is None:
        tokenizer = FakeTokenizer()
    kwargs = dict(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        enable_cuda_graph=enable_cuda_graph,
        enable_overlap=enable_overlap,
        backend="torch_native",
    )
    if device is not None:
        kwargs["device"] = device
    kwargs.update(overrides)
    return Scheduler(model=model, tokenizer=tokenizer, **kwargs)


def make_cpu_model(max_position_embeddings: int = 64):
    """Tiny deterministic-sequence model for scheduler-pipeline tests."""
    return AutoRegressiveLM(
        make_rollout_config(max_position_embeddings=max_position_embeddings)
    ).eval()


@pytest.fixture(autouse=True)
def _cleanup_app_engine():
    """Reset the lazy FastAPI singleton engine after each inference test."""
    yield
    get_app().state.engine = None


@pytest.fixture
def client():
    """Provide a test client for the FastAPI app."""
    _app = get_app()
    _app.state.server_config = {
        "device": "cpu",
        "dtype": "bfloat16",
        "param_path": None,
        "max_batch_size": 1,
        "_test": True,
    }
    _app.state.engine = None
    return TestClient(_app)


@pytest.fixture
def mock_engine():
    """Create a mock InferenceEngine."""

    async def _async_gen():
        yield "chunk1"
        yield "chunk2"
        yield "[DONE]"

    mock = MagicMock()
    mock.generate.return_value = "mock response"
    mock.generate_async.return_value = _async_gen()
    mock.get_stats.return_value = {
        "total_tasks": 0,
        "total_tokens": 0,
        "running": 0,
        "waiting": 0,
    }
    mock.tokenizer.encode.return_value = [1, 2, 3]
    mock.tokenizer.decode.return_value = "mock response"
    mock.tokenizer.apply_chat_template.return_value = "mock prompt"
    return mock


@pytest.fixture
def loaded_model(client, mock_engine):
    """Simulate that the engine is loaded."""
    get_app().state.engine = mock_engine
    return mock_engine
