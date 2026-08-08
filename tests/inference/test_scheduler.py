"""Tests for scheduler concurrency."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from astrai.inference import InferenceScheduler
from astrai.inference.runtime.executor import DecodeSteadyState, Executor
from astrai.inference.task import Task
from astrai.model.transformer import AutoRegressiveLM
from tests.helpers import FakeTokenizer, make_rollout_config


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.config.num_key_value_heads = 8
    mock_model.config.num_attention_heads = 8
    mock_model.config.hidden_size = 128
    mock_model.config.num_hidden_layers = 2
    mock_model.config.max_position_embeddings = 100
    mock_model.parameters.return_value = iter(
        [MagicMock(dtype=torch.float32, device=torch.device("cpu"))]
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "token"
    mock_tokenizer.stop_ids = [0]
    mock_tokenizer.pad_id = None

    return mock_model, mock_tokenizer


def test_scheduler_concurrent_add_task(mock_model_and_tokenizer):
    """Test concurrent add_task operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.scheduler.AutoModel"):
        with patch("astrai.inference.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    results = {"task_ids": [], "errors": []}
    lock = threading.Lock()

    def add_task_worker(worker_id):
        try:
            for i in range(10):
                task_id = scheduler.add_task(f"prompt from worker {worker_id}-{i}")
                with lock:
                    results["task_ids"].append(task_id)
        except Exception as e:
            results["errors"].append(str(e))

    threads = [threading.Thread(target=add_task_worker, args=(i,)) for i in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    scheduler.stop()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["task_ids"]) == 50


def test_scheduler_concurrent_add_remove_task(mock_model_and_tokenizer):
    """Test concurrent add and remove task operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.scheduler.AutoModel"):
        with patch("astrai.inference.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    results = {"added": [], "removed": [], "errors": []}
    add_ready = threading.Event()

    def add_worker():
        try:
            for i in range(20):
                task_id = scheduler.add_task(f"prompt {i}")
                results["added"].append(task_id)
                if len(results["added"]) >= 10:
                    add_ready.set()
        except Exception as e:
            results["errors"].append(f"Add: {str(e)}")

    def remove_worker():
        try:
            add_ready.wait(timeout=5.0)
            for task_id in results["added"][:10]:
                scheduler.remove_task(task_id)
                results["removed"].append(task_id)
        except Exception as e:
            results["errors"].append(f"Remove: {str(e)}")

    add_thread = threading.Thread(target=add_worker)
    remove_thread = threading.Thread(target=remove_worker)

    add_thread.start()
    remove_thread.start()

    add_thread.join()
    remove_thread.join()
    scheduler.stop()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["added"]) == 20


def test_scheduler_concurrent_get_stats(mock_model_and_tokenizer):
    """Test concurrent get_stats operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.scheduler.AutoModel"):
        with patch("astrai.inference.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    results = {"stats": [], "errors": []}
    started = threading.Event()
    stats_done = threading.Event()

    def add_tasks():
        try:
            for i in range(20):
                scheduler.add_task(f"prompt {i}")
                started.set()
        except Exception as e:
            results["errors"].append(f"Add: {str(e)}")

    def get_stats():
        try:
            started.wait(timeout=5.0)
            for _ in range(50):
                stats = scheduler.get_stats()
                results["stats"].append(stats)
            stats_done.set()
        except Exception as e:
            results["errors"].append(f"Get stats: {str(e)}")

    add_thread = threading.Thread(target=add_tasks)
    stats_thread = threading.Thread(target=get_stats)

    add_thread.start()
    stats_thread.start()

    add_thread.join()
    stats_done.wait(timeout=5.0)
    scheduler.stop()

    stats_thread.join()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["stats"]) == 50

    for stats in results["stats"]:
        assert "total_tasks" in stats
        assert stats["total_tasks"] >= 0


def _make_real_scheduler(device):
    """Build a scheduler backed by a tiny real model for run_batch tests."""
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = FakeTokenizer()
    scheduler = InferenceScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
    )
    return scheduler, tokenizer, model


def test_run_batch_returns_token_sequences(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30], [5, 6, 7, 8]]
        results = scheduler.run_batch(prompts, max_tokens=4, temperature=1.0)
        assert len(results) == 2
        for ids in results:
            assert isinstance(ids, list)
            assert len(ids) <= 4
            assert all(0 <= i < 200 for i in ids)
    finally:
        scheduler.stop()


def test_run_batch_tokens_match_full_sequence_forward(device):
    scheduler, _tok, model = _make_real_scheduler(device)
    prompt = [10, 20, 30, 40]
    try:
        expected = []
        sequence = list(prompt)
        for _ in range(2):
            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            position_ids = torch.arange(len(sequence), device=device).unsqueeze(0)
            input_mask = torch.ones(
                1, len(sequence), len(sequence), dtype=torch.bool, device=device
            ).tril()
            with torch.inference_mode():
                logits = model(
                    input_ids,
                    input_mask=input_mask,
                    position_ids=position_ids,
                )["logits"][:, -1, :]
            token = logits.argmax(dim=-1).item()
            expected.append(token)
            sequence.append(token)

        result = scheduler.run_batch(
            prompt_ids_list=[prompt], max_tokens=2, temperature=0
        )
        assert result == [expected]
    finally:
        scheduler.stop()


def test_run_batch_return_logprobs_aligned(device):
    """return_logprobs=True gives (token_ids, logprobs) tuples with equal len."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30, 40]]
        results = scheduler.run_batch(
            prompts, max_tokens=5, temperature=1.0, return_logprobs=True
        )
        assert len(results) == 1
        token_ids, logprobs = results[0]
        assert len(token_ids) == len(logprobs)
        assert all(lp <= 1e-5 for lp in logprobs)  # logprobs ≤ 0
    finally:
        scheduler.stop()


def test_run_batch_respects_max_tokens(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30]]
        results = scheduler.run_batch(prompts, max_tokens=3, temperature=1.0)
        assert len(results[0]) <= 3
    finally:
        scheduler.stop()


def test_run_batch_zero_max_tokens_returns_empty(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.run_batch([[10, 20, 30]], max_tokens=0) == [[]]
    finally:
        scheduler.stop()


def test_run_batch_stop_id_terminates(device):
    """A token matching stop_ids terminates generation for that prompt."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30]]
        results = scheduler.run_batch(prompts, max_tokens=32, temperature=1.0)
        # If stop token 2 was produced, it is the last token
        if results[0] and results[0][-1] == 2:
            # No tokens after stop should exist (since we terminate)
            assert 2 not in results[0][:-1]
    finally:
        scheduler.stop()


def test_run_batch_empty_prompts(device):
    """Empty prompt list yields empty result list."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.run_batch([], max_tokens=4) == []
    finally:
        scheduler.stop()


def test_run_batch_too_long_prompt_skipped(device):
    """A prompt longer than max_seq_len yields an empty result slot."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        long = list(range(100))  # > max_seq_len=64
        results = scheduler.run_batch([long, [10, 20]], max_tokens=2)
        assert results[0] == []
        assert len(results[1]) <= 2
    finally:
        scheduler.stop()


def test_decode_does_not_reuse_previous_batch_state():
    executor = object.__new__(Executor)
    executor.device = torch.device("cpu")
    executor.task_cache = MagicMock()
    executor.task_cache.bind_was_steady = True
    executor.task_cache.bind.return_value = MagicMock()
    executor._graph_supported = False
    executor._graph_ctx = SimpleNamespace(enabled=False)

    workspace = MagicMock()
    workspace.position_ids = torch.tensor([2], dtype=torch.long)
    workspace.fill_input_ids.return_value = torch.tensor([7], dtype=torch.long)
    workspace.decode_mask.return_value = torch.ones(1, 1, 9, dtype=torch.bool)
    executor._workspace = workspace
    executor.model = MagicMock(
        return_value={"logits": torch.zeros(1, 1, 10, dtype=torch.float32)}
    )

    old_info = object()
    new_info = object()
    executor._decode_cache = DecodeSteadyState(("old",), [2], old_info)
    executor._sample_logits = MagicMock(return_value=[3])

    task = Task("new", list(range(8)), temperature=0)
    task.input_tokens = 8
    task.output_ids = [7]
    task.mark_prefill_done()

    with patch(
        "astrai.inference.runtime.executor._build_sampling_batch_info",
        return_value=new_info,
    ):
        assert executor.execute_decode([task]) == [3]

    assert workspace.position_ids.tolist() == [8]
    assert executor._decode_cache.task_sig == ("new",)
    executor._sample_logits.assert_called_once()
    args, kwargs = executor._sample_logits.call_args
    assert args[1:] == ([task], False)
    assert kwargs["info"] is new_info
