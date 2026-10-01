"""Tests for scheduler concurrency."""

import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from astrai.extension import CudaBackend, TorchNativeBackend, get_backend
from astrai.inference import GenerationResult, Scheduler
from astrai.inference.core.metrics import MetricsCollector
from astrai.inference.core.request import STOP, BatchedStreamCallback, Request
from astrai.inference.core.stepper import SchedulerStep
from astrai.inference.worker.model_runner import (
    DecodeSteadyState,
    GPUModelRunner,
)
from astrai.inference.worker.pending import (
    BatchSnapshot,
    PendingExecution,
    ResultRing,
)
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


def _make_mock_scheduler(mock_model_and_tokenizer):
    """Build a CPU scheduler over mocks, patching scheduler-internal imports."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    with (
        patch("astrai.inference.core.scheduler.AutoModel"),
        patch("astrai.inference.core.scheduler.AutoTokenizer"),
    ):
        return Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            max_batch_size=4,
            device="cpu",
        )


def _run_threads(*workers, timeout=10.0):
    threads = [threading.Thread(target=worker) for worker in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)


def test_scheduler_concurrent_add_task(mock_model_and_tokenizer):
    """Test concurrent add_request operations."""
    scheduler = _make_mock_scheduler(mock_model_and_tokenizer)

    results = {"request_ids": [], "errors": []}
    lock = threading.Lock()

    def add_task_worker(worker_id):
        try:
            for i in range(10):
                request_id = scheduler.add_request(
                    f"prompt from worker {worker_id}-{i}"
                )
                with lock:
                    results["request_ids"].append(request_id)
        except Exception as e:
            results["errors"].append(str(e))

    _run_threads(*(lambda wid=i: add_task_worker(wid) for i in range(5)))

    scheduler.stop()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["request_ids"]) == 50


def test_generation_loop_activates_backend_in_worker_thread():
    scheduler = object.__new__(Scheduler)
    scheduler._backend = TorchNativeBackend()
    scheduler._stop_event = threading.Event()
    scheduler._kv_manager = MagicMock()
    scheduler._retired = []
    scheduler._executor = MagicMock()
    scheduler._executor.peek_pending.return_value = None
    scheduler._stepper = MagicMock()

    observed = []
    task_mgr = MagicMock()
    task_mgr.tokenizer.stop_ids = [0]
    task_mgr.remove_finished_requests.return_value = []
    task_mgr.get_running_requests.return_value = []
    task_mgr.max_batch_size = 1
    task_mgr.pull_waiting.return_value = []
    task_mgr.has_requests.return_value = False

    def observe_backend(*args, **kwargs):
        observed.append(type(get_backend()))
        scheduler._stop_event.set()

    task_mgr.wait_for_requests.side_effect = observe_backend
    scheduler._requests = task_mgr

    thread = threading.Thread(target=scheduler.run_busy_loop)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert observed == [TorchNativeBackend]


def test_step_splits_decode_batch_by_request_backend():
    scheduler = object.__new__(Scheduler)
    scheduler._cache = SimpleNamespace(page_size=1)
    scheduler._kv_manager = MagicMock()
    scheduler._kv_manager.extend_slots.return_value = True
    scheduler._kv_manager.extend_slots_batch.return_value = [True, True]
    scheduler._metrics = MetricsCollector()
    scheduler._executor = MagicMock()
    scheduler._stepper = SchedulerStep(
        scheduler._cache, scheduler._kv_manager, scheduler._executor, scheduler._metrics
    )

    observed = []

    def submit(*args, **kwargs):
        requests = args[0]
        observed.append(
            (type(get_backend()), [request.request_id for request in requests])
        )
        return PendingExecution(
            snapshot=BatchSnapshot(
                request_ids=tuple(t.request_id for t in requests),
                kv_positions=(0,),
                policy_version=0,
            ),
            requests=list(requests),
            tokens=torch.tensor([1] * len(requests), dtype=torch.long),
        )

    scheduler._executor.submit_decode.side_effect = submit

    torch_task = Request("torch", [1], backend=TorchNativeBackend())
    cuda_task = Request("cuda", [1], backend=CudaBackend())
    for request in (torch_task, cuda_task):
        request.input_tokens = 1
        request.output_ids = [1]
        request.mark_prefill_complete()
        scheduler._metrics.register(request.request_id)

    produced, aborted = scheduler._step([torch_task, cuda_task])

    assert aborted == []
    assert produced == [torch_task, cuda_task]
    assert observed == [
        (TorchNativeBackend, ["torch"]),
        (CudaBackend, ["cuda"]),
    ]


def test_step_batches_ragged_prefill_with_shared_cache_start():
    scheduler = object.__new__(Scheduler)
    scheduler._cache = SimpleNamespace(page_size=64)
    scheduler._kv_manager = MagicMock()
    scheduler._kv_manager.cached_tokens.return_value = 0
    scheduler._metrics = MetricsCollector()
    scheduler._executor = MagicMock()
    scheduler._stepper = SchedulerStep(
        scheduler._cache, scheduler._kv_manager, scheduler._executor, scheduler._metrics
    )

    short = Request("short", [1, 2, 3])
    long = Request("long", [4, 5, 6, 7, 8])
    for request in (short, long):
        scheduler._metrics.register(request.request_id)

    scheduler._executor.execute_prefill.return_value = (
        [long, short],
        PendingExecution(
            snapshot=BatchSnapshot(("long", "short"), (0,), 0),
            requests=[long, short],
            tokens=torch.tensor([11, 12], dtype=torch.long),
        ),
    )

    produced, aborted = scheduler._step([short, long])

    assert aborted == []
    # ``produced`` is now the live set (callers drive it): same members,
    # caller order, not just the requests that sampled this step.
    assert set(produced) == {long, short}
    assert produced == [short, long]
    scheduler._executor.execute_prefill.assert_called_once_with(
        [short, long],
        start_pos=0,
        return_logprobs=False,
        num_tokens=[3, 5],
    )
    assert long.output_ids == [11]
    assert short.output_ids == [12]


def test_execute_prefill_packs_ragged_prompts_and_selects_last_logits():
    executor = object.__new__(GPUModelRunner)
    executor.device = torch.device("cpu")
    executor.kv_manager = MagicMock()
    executor.kv_manager.bind.return_value = MagicMock()
    executor._workspace = MagicMock()
    executor._workspace.max_batch_size = 16  # Add max_batch_size for validation
    all_logits = torch.arange(42, dtype=torch.float32).reshape(6, 7)

    def fake_model(ids, *, position_ids, kv_cache, fwd, logits_positions):
        return {"logits": all_logits[logits_positions]}

    executor.model = MagicMock(side_effect=fake_model)
    executor._submit_sample = MagicMock(
        return_value=PendingExecution(
            snapshot=BatchSnapshot(("a", "b"), (0,), 0),
            requests=[],
            tokens=torch.tensor([101, 102]),
        )
    )

    task_b = Request("b", [20, 21, 22, 23, 24])
    task_a = Request("a", [10, 11, 12])

    requests, pending = executor.execute_prefill([task_b, task_a], start_pos=1)

    assert requests == [task_a, task_b]
    assert pending.commit() == [(101, None), (102, None)]
    model_args, model_kwargs = executor.model.call_args
    assert model_args[0].tolist() == [11, 12, 21, 22, 23, 24]
    assert model_kwargs["position_ids"].tolist() == [1, 2, 1, 2, 3, 4]
    assert model_kwargs["logits_positions"].tolist() == [1, 5]
    executor.kv_manager.bind.assert_called_once_with(
        ["a", "b"], executor._workspace, start_pos=1, seq_ends=[3, 5]
    )
    sample_args, sample_kwargs = executor._submit_sample.call_args
    torch.testing.assert_close(sample_args[0], all_logits[[1, 5]])
    assert sample_args[1] == [task_a, task_b]
    assert sample_args[2] is False
    assert sample_kwargs == {}


def test_scheduler_concurrent_add_remove_task(mock_model_and_tokenizer):
    """Test concurrent add and remove request operations."""
    scheduler = _make_mock_scheduler(mock_model_and_tokenizer)

    results = {"added": [], "removed": [], "errors": []}
    add_ready = threading.Event()

    def add_worker():
        try:
            for i in range(20):
                request_id = scheduler.add_request(f"prompt {i}")
                results["added"].append(request_id)
                if len(results["added"]) >= 10:
                    add_ready.set()
        except Exception as e:
            results["errors"].append(f"Add: {str(e)}")

    def remove_worker():
        try:
            add_ready.wait(timeout=5.0)
            for request_id in results["added"][:10]:
                scheduler.remove_request(request_id)
                results["removed"].append(request_id)
        except Exception as e:
            results["errors"].append(f"Remove: {str(e)}")

    _run_threads(add_worker, remove_worker)
    scheduler.stop()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["added"]) == 20


def test_scheduler_concurrent_get_stats(mock_model_and_tokenizer):
    """Test concurrent get_stats operations."""
    scheduler = _make_mock_scheduler(mock_model_and_tokenizer)

    results = {"stats": [], "errors": []}
    started = threading.Event()
    stats_done = threading.Event()

    def add_requests():
        try:
            for i in range(20):
                scheduler.add_request(f"prompt {i}")
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

    _run_threads(add_requests, get_stats)
    scheduler.stop()
    stats_done.wait(timeout=5.0)

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
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
    )
    return scheduler, tokenizer, model


def test_cancel_waiting_task_storm_returns_to_baseline(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        request_ids = [
            scheduler.add_request(f"waiting-{index}", max_tokens=32)
            for index in range(32)
        ]

        assert all(scheduler.cancel_request(request_id) for request_id in request_ids)
        stats = scheduler.get_stats()
        assert stats["running"] == 0
        assert stats["waiting_tasks"] == 0
        assert stats["in_flight_tasks"] == 0
        assert stats["kv_cache_tasks"] == 0
        assert stats["cancelled_total"] == len(request_ids)
    finally:
        scheduler.stop()


def test_cancel_active_task_releases_metrics_and_kv(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        request_id = scheduler.add_request("active", max_tokens=32)
        request = scheduler._requests.pull_waiting(1)[0]
        assert scheduler._kv_manager.alloc_slots(request.request_id, request.prompt_ids)
        assert scheduler._requests.activate(request)

        before = scheduler.get_stats()
        assert before["running"] == 1
        assert before["in_flight_tasks"] == 1
        assert before["kv_cache_tasks"] == 1

        assert scheduler.cancel_request(request_id)
        scheduler.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            after = scheduler.get_stats()
            if (
                after["running"] == 0
                and after["in_flight_tasks"] == 0
                and after["kv_cache_tasks"] == 0
            ):
                break
            time.sleep(0.01)

        assert after["running"] == 0
        assert after["waiting_tasks"] == 0
        assert after["in_flight_tasks"] == 0
        assert after["kv_cache_tasks"] == 0
        assert after["cancelled_total"] == 1
    finally:
        scheduler.stop()


def test_cancel_during_kv_allocation_releases_metrics_and_kv(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    allocation_started = threading.Event()
    continue_allocation = threading.Event()
    original_alloc = scheduler._kv_manager.alloc_slots

    def blocking_alloc(*args, **kwargs):
        allocation_started.set()
        assert continue_allocation.wait(timeout=5)
        return original_alloc(*args, **kwargs)

    try:
        with patch.object(
            scheduler._kv_manager,
            "alloc_slots",
            side_effect=blocking_alloc,
        ):
            scheduler.start()
            request_id = scheduler.add_request("allocation-race", max_tokens=32)
            assert allocation_started.wait(timeout=5)
            assert scheduler.cancel_request(request_id)
            continue_allocation.set()

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                stats = scheduler.get_stats()
                if (
                    stats["running"] == 0
                    and stats["waiting_tasks"] == 0
                    and stats["in_flight_tasks"] == 0
                    and stats["kv_cache_tasks"] == 0
                ):
                    break
                time.sleep(0.01)

        assert stats["running"] == 0
        assert stats["waiting_tasks"] == 0
        assert stats["in_flight_tasks"] == 0
        assert stats["kv_cache_tasks"] == 0
        assert stats["cancelled_total"] == 1
    finally:
        continue_allocation.set()
        scheduler.stop()


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


def test_ragged_prefill_matches_sequential_greedy_tokens_and_logprobs(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    prompts = [
        [10, 20, 30],
        [5, 6, 7, 8],
        [40, 41, 42, 43, 44],
    ]
    try:
        ragged = scheduler.run_batch(
            prompts, max_tokens=1, temperature=0, return_logprobs=True
        )
        sequential = [
            scheduler.run_batch(
                [prompt], max_tokens=1, temperature=0, return_logprobs=True
            )[0]
            for prompt in prompts
        ]

        assert [result[0] for result in ragged] == [result[0] for result in sequential]
        for ragged_result, sequential_result in zip(ragged, sequential):
            assert ragged_result[1] == pytest.approx(sequential_result[1], abs=1e-6)
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
        # Make every token a stop id: generation must end after exactly
        # one token (the stop token itself) instead of running to max_tokens.
        scheduler._requests.tokenizer.stop_ids = list(range(200))
        prompts = [[10, 20, 30]]
        results = scheduler.run_batch(prompts, max_tokens=32, temperature=1.0)
        assert len(results[0]) == 1
    finally:
        scheduler.stop()


def test_run_batch_empty_prompts(device):
    """Empty prompt list yields empty result list."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.run_batch([], max_tokens=4) == []
    finally:
        scheduler.stop()


def test_scheduler_weight_versions_are_monotonic_and_acknowledged(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.policy_version == 0
        assert scheduler.update_weights(1) == 1
        assert scheduler.policy_version == 1
        assert scheduler.get_stats()["policy_version"] == 1
        assert scheduler.update_weights(1) == 1
        with pytest.raises(ValueError, match="cannot move backwards"):
            scheduler.update_weights(0)
        with pytest.raises(ValueError, match="non-negative integer"):
            scheduler.update_weights(True)
    finally:
        scheduler.stop()


def test_scheduler_applies_weight_mutation_and_version_atomically(device):
    scheduler, _tok, model = _make_real_scheduler(device)
    before = next(model.parameters()).detach().clone()

    def mutate(policy_version):
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return "updated"

    try:
        assert scheduler.apply_weight_update(1, mutate) == "updated"
        assert scheduler.policy_version == 1
        assert not torch.equal(next(model.parameters()), before)
        with pytest.raises(ValueError, match="must advance"):
            scheduler.apply_weight_update(1, mutate)

        def failed_mutation(policy_version):
            raise RuntimeError("optimizer failed")

        with pytest.raises(RuntimeError, match="optimizer failed"):
            scheduler.apply_weight_update(2, failed_mutation)
        assert scheduler.policy_version == 1

        # None derives live+1 under the lock: no read-compute-write race
        # on the current version for advance-by-one callers. The derived
        # target version is handed to the update callable.
        seen_versions = []

        def record(policy_version):
            seen_versions.append(policy_version)
            return "updated"

        assert scheduler.apply_weight_update(None, record) == "updated"
        assert scheduler.policy_version == 2
        assert seen_versions == [2]
    finally:
        scheduler.stop()


def test_scheduler_atomic_advance_survives_interleaved_publish(device):
    """A concurrent publish between reading the live version and applying
    the update must not fail ``require_advance`` (regression: callers
    computed live+1 outside the lock, a TOCTOU that raised spuriously)."""
    scheduler, _tok, _model = _make_real_scheduler(device)

    try:
        # Simulate the race directly: a version read that goes stale before
        # apply_weight_update acquires the lock. With None the scheduler
        # re-derives live+1 inside the critical section.
        stale_read = scheduler.policy_version + 1
        scheduler.update_weights(1)
        assert stale_read == 1  # now equals live -> explicit form would raise
        with pytest.raises(ValueError, match="must advance"):
            scheduler.apply_weight_update(stale_read, lambda _version: "ok")
        assert scheduler.apply_weight_update(None, lambda _version: "ok") == "ok"
        assert scheduler.policy_version == 2
    finally:
        scheduler.stop()


def test_scheduler_serializes_policy_snapshot_and_direct_update(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    update_finished = threading.Event()
    errors = []

    def inspect(version):
        assert version == 0
        snapshot_started.set()
        assert release_snapshot.wait(timeout=5)

    def take_snapshot():
        try:
            scheduler.with_policy_snapshot(inspect)
        except BaseException as exc:
            errors.append(exc)

    def update():
        try:
            scheduler.update_weights(1)
            update_finished.set()
        except BaseException as exc:
            errors.append(exc)

    snapshot_thread = threading.Thread(target=take_snapshot)
    update_thread = threading.Thread(target=update)
    try:
        snapshot_thread.start()
        assert snapshot_started.wait(timeout=5)
        update_thread.start()
        assert not update_finished.wait(timeout=0.1)
        release_snapshot.set()
        snapshot_thread.join(timeout=5)
        update_thread.join(timeout=5)
        assert not snapshot_thread.is_alive()
        assert not update_thread.is_alive()
        assert errors == []
        assert scheduler.policy_version == 1
    finally:
        release_snapshot.set()
        snapshot_thread.join(timeout=5)
        update_thread.join(timeout=5)
        scheduler.stop()


def test_scheduler_rejects_weight_update_with_queued_tasks(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    request_id = scheduler.add_request("queued")
    try:
        with pytest.raises(RuntimeError, match="while requests are queued"):
            scheduler.update_weights(1)
        scheduler.remove_request(request_id)
        assert scheduler.update_weights(1) == 1
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


def test_run_batch_details_distinguish_rejection_from_success(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        long_prompt = list(range(100))
        results = scheduler.run_batch(
            [long_prompt, [10, 20]],
            max_tokens=2,
            temperature=0,
            return_logprobs=True,
            return_details=True,
        )

        assert results[0] == GenerationResult(
            token_ids=[],
            logprobs=[],
            finish_reason="rejected",
            error_reason="prompt_too_long",
        )
        assert results[1].finish_reason in ("stop", "length")
        assert results[1].error_reason is None
        assert len(results[1].token_ids) == len(results[1].logprobs)
    finally:
        scheduler.stop()


def test_run_batch_details_report_non_positive_max_tokens(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        result = scheduler.run_batch([[10, 20]], max_tokens=0, return_details=True)[0]
        assert result.finish_reason == "rejected"
        assert result.error_reason == "max_tokens_non_positive"
    finally:
        scheduler.stop()


def test_run_batch_details_report_allocation_failure(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        with patch.object(scheduler._kv_manager, "alloc_slots", return_value=False):
            result = scheduler.run_batch([[10, 20]], max_tokens=2, return_details=True)[
                0
            ]
        assert result.finish_reason == "rejected"
        assert result.error_reason == "kv_cache_allocation_failed"
    finally:
        scheduler.stop()


def test_run_batch_details_report_extension_failure_and_cleanup(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        with patch.object(
            scheduler._stepper,
            "step",
            side_effect=lambda requests, **_kwargs: ([], list(requests)),
        ):
            result = scheduler.run_batch([[10, 20]], max_tokens=2, return_details=True)[
                0
            ]

        assert result.finish_reason == "rejected"
        assert result.error_reason == "kv_cache_extension_failed"
        assert scheduler._kv_manager._states == {}
        assert scheduler._metrics._timings == {}
    finally:
        scheduler.stop()


def test_decode_does_not_reuse_previous_batch_state():
    executor = object.__new__(GPUModelRunner)
    executor.device = torch.device("cpu")
    executor.kv_manager = MagicMock()
    executor.kv_manager.bind_was_steady = True
    executor.kv_manager.bind.return_value = MagicMock()
    executor._graph_supported = False
    executor._graph_ctx = SimpleNamespace(enabled=False)
    executor._pending = None
    executor._result_ring = ResultRing(16, executor.device)

    workspace = MagicMock()
    workspace.max_batch_size = 16
    workspace.position_ids = torch.tensor([2], dtype=torch.long)
    workspace.fill_input_ids.return_value = torch.tensor([7], dtype=torch.long)
    workspace.decode_mask.return_value = torch.ones(1, 1, 9, dtype=torch.bool)
    executor._workspace = workspace
    executor.model = MagicMock(
        return_value={"logits": torch.zeros(1, 1, 10, dtype=torch.float32)}
    )

    old_info = object()
    new_info = SimpleNamespace(has_freq=False)
    executor._decode_cache = DecodeSteadyState(("old",), [2], old_info)
    pending_result = PendingExecution(
        snapshot=BatchSnapshot(("new",), (8,), 0),
        requests=[],
        tokens=torch.tensor([3], dtype=torch.long),
    )
    executor._submit_sample = MagicMock(return_value=pending_result)

    request = Request("new", list(range(8)), temperature=0)
    request.input_tokens = 8
    request.output_ids = [7]
    request.mark_prefill_complete()

    with patch(
        "astrai.inference.worker.model_runner._build_sampling_batch_info",
        return_value=new_info,
    ):
        assert executor.execute_decode([request]) == [3]

    assert workspace.position_ids.tolist() == [8]
    assert executor._decode_cache.task_sig == ("new",)
    executor._submit_sample.assert_called_once()
    args, kwargs = executor._submit_sample.call_args
    assert args[1:] == ([request], False)
    assert kwargs["info"] is new_info


def test_decode_fills_input_ids_from_device_on_matching_signature():
    """Steady-state decode copies cached device tokens, skipping the host."""
    executor = object.__new__(GPUModelRunner)
    executor.device = torch.device("cpu")
    executor.kv_manager = MagicMock()
    executor.kv_manager.bind_was_steady = True
    executor.kv_manager.bind.return_value = MagicMock()
    executor._graph_supported = False
    executor._graph_ctx = SimpleNamespace(enabled=False)

    workspace = MagicMock()
    workspace.max_batch_size = 16
    workspace.position_ids = torch.tensor([2], dtype=torch.long)
    workspace.fill_input_ids_from_device.return_value = torch.tensor(
        [9], dtype=torch.long
    )
    executor._workspace = workspace
    executor.model = MagicMock(
        return_value={"logits": torch.zeros(1, 1, 10, dtype=torch.float32)}
    )

    info = SimpleNamespace(has_freq=False)
    tokens = torch.tensor([3], dtype=torch.long)
    executor._decode_cache = DecodeSteadyState(("t1",), [2], info, last_tokens=tokens)
    executor._pending = None
    executor._result_ring = ResultRing(16, executor.device)
    executor._submit_sample = MagicMock(
        return_value=PendingExecution(
            snapshot=BatchSnapshot(("t1",), (3,), 0), requests=[], tokens=tokens
        )
    )

    request = Request("t1", list(range(8)), temperature=0)
    request.input_tokens = 8
    request.output_ids = [7]
    request.mark_prefill_complete()

    with patch(
        "astrai.inference.worker.model_runner._build_sampling_batch_info",
        return_value=info,
    ):
        assert executor.execute_decode([request]) == [3]

    workspace.fill_input_ids.assert_not_called()
    workspace.fill_input_ids_from_device.assert_called_once_with(tokens)
    assert workspace.position_ids.tolist() == [3]
    assert executor._decode_cache.task_sig == ("t1",)
    assert executor._decode_cache.last_tokens is tokens


class _MultiPatch:
    def __init__(self, patches):
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


def test_steady_decode_submit_path_is_sync_free(device):
    """Gate: no hidden device-to-host syncs on the steady decode hot path.

    After the first decode step of an unchanged batch (the steady state the
    serving loop lives in), one ``execute_decode`` call must not resolve any
    device predicate or scalar to host: ``Tensor.item``/``any``/``all``
    probes are patched to raise, and only ``tolist`` stays legal (it is
    the single intentional commit point that yields the sampled tokens).
    """
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30, 40], [5, 6, 7, 8]]
        # First run_batch call warms everything the steady state touches
        # (prefill, first decode, sampling-info build) so the instrumented
        # second batch enters steady state from its first step.
        scheduler.run_batch(prompts, max_tokens=2, temperature=0.7, top_k=10)
        torch.manual_seed(1234)

        banned = (
            torch.Tensor.item,
            torch.Tensor.any,
            torch.Tensor.all,
        )
        calls: list = []

        def _spy(name, orig):
            def _impl(self, *a, **k):
                calls.append(name)
                return orig(self, *a, **k)

            return _impl

        patched = tuple(
            patch.object(torch.Tensor, name, _spy(name, orig))
            for name, orig in zip(("item", "any", "all"), banned)
        )
        with _MultiPatch(patched):
            scheduler.run_batch(prompts, max_tokens=2, temperature=0.7, top_k=10)
        assert not calls, f"steady decode resolved device predicates on host: {calls}"
    finally:
        scheduler.stop()


def test_run_batch_greedy_reproducible_across_calls(device):
    """Greedy decode is bit-identical across invocations of one scheduler.

    Guards the submit/commit split without RNG in play: argmax tokens must
    be stable when the same batch runs twice through the same engine.

    Stochastic reproducibility across run_batch calls is deliberately NOT
    asserted: bf16 GEMM reductions are not bit-deterministic under
    different kernel interleavings, and the async result relay changes the
    host/GPU overlap between calls. Same-seed replay is only guaranteed
    within identical execution conditions.
    """
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30, 40], [7, 8, 9]]
        first = scheduler.run_batch(prompts, max_tokens=6, temperature=0)
        second = scheduler.run_batch(prompts, max_tokens=6, temperature=0)
        assert first == second
        assert all(len(ids) == 6 for ids in first)
    finally:
        scheduler.stop()


def test_pending_step_commit_is_idempotent(device):
    """A committed step cannot double-append tokens to its requests."""
    tokens = torch.tensor([5, 6], dtype=torch.long)
    logprobs = torch.tensor([-1.5, -2.5], dtype=torch.float32)
    pending = PendingExecution(
        snapshot=BatchSnapshot(("a", "b"), (0, 0), 0),
        requests=[object(), object()],
        tokens=tokens,
        logprobs=logprobs,
    )
    first = pending.commit()
    second = pending.commit()
    assert first == second
    assert first == [(5, -1.5), (6, -2.5)]
    assert pending.committed


def test_submit_decode_returns_pending_without_touching_tasks(device):
    """The submit half leaves request output state untouched.

    Core of the B1 contract: after submit, no token is appended — the
    scheduler may still roll the batch back.  (The KV write cursor is a
    submit-side property — see SchedulerStep._submit_decoded — because the
    launched work owns its write slot; commit only materialises results.)
    """
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30, 40]]
        scheduler.run_batch(prompts, max_tokens=2, temperature=0.7)
        requests = [
            Request(
                request_id=f"probe_{uuid.uuid4().hex[:8]}",
                prompt_ids=[10, 20, 30, 40],
                max_tokens=4,
                temperature=0.7,
            )
        ]
        request = requests[0]
        if not scheduler._kv_manager.alloc_slots(
            request.request_id, request.prompt_ids
        ):
            pytest.skip("KV allocation failed")
        request.input_tokens = len(request.prompt_ids)
        # Prefill through the real stepper path so KV state is complete.
        with scheduler._backend_context():
            scheduler._stepper.step([request])
            assert request.prefill_complete
            pending = scheduler._executor.submit_decode([request])
        assert pending is not None
        try:
            n_prefilled = len(request.output_ids)
            assert request.output_tokens == n_prefilled
            scheduler._stepper.step_commit(pending)
            assert len(request.output_ids) == n_prefilled + 1
            assert request.output_tokens == n_prefilled + 1
        finally:
            scheduler._kv_manager.free_slots(request.request_id)
    finally:
        scheduler.stop()


def test_online_loop_overlap_generates_and_admits_midstream(device):
    """End-to-end online generation rides the overlap pipeline.

    A real model serves one request through the scheduler thread, then a
    second request joins mid-decode.  The overlap branch (submit current,
    commit previous) must deliver every token of both requests in order,
    handle the mid-run batch change through its drain-and-sync fallback,
    and leave no in-flight step behind at stop.
    """
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    tok = FakeTokenizer()
    scheduler = Scheduler(
        model=model,
        tokenizer=tok,
        max_batch_size=8,
        max_seq_len=64,
        enable_overlap=True,
    )

    # FakeTokenizer.encode returns the batch shape ([[ids]]); add_request's
    # contract is the flat single-string shape, so unwrap it here.
    class _UnwrappingTokenizer:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "stop_ids", inner.stop_ids)

        def encode(self, prompt, **kw):
            out = self._inner.encode(prompt, **kw)
            return out[0] if isinstance(out[0], list) else out

        def __getattr__(self, name):
            return getattr(self._inner, name)

    scheduler._requests.tokenizer = _UnwrappingTokenizer(tok)

    # StreamDecoder requires the Rust tokenizer's ``_tokenizer`` handle,
    # which FakeTokenizer does not provide; the online e2e only needs
    # per-token text events, so substitute a trivial decoder.
    class _FakeStreamDecoder:
        def __init__(self, tokenizer):
            pass

        def push(self, token_id):
            return f"<{token_id}>" if token_id > 2 else ""

    events: dict = {"first": [], "second": []}
    done: dict = {"first": threading.Event(), "second": threading.Event()}

    def make_sink(tag):
        class _Sink(BatchedStreamCallback):
            def __call__(self, batch):
                for request_id, token in batch:
                    events[tag].append(token)
                    if token is STOP:
                        done[tag].set()

        return _Sink()

    try:
        with patch("astrai.inference.core.request.StreamDecoder", _FakeStreamDecoder):
            scheduler.start()
            scheduler.add_request(
                "a" * 8, max_tokens=12, stream_callback=make_sink("first")
            )
            time.sleep(0.2)  # let the first request enter steady decode
            scheduler.add_request(
                "b" * 8, max_tokens=12, stream_callback=make_sink("second")
            )
            assert done["first"].wait(timeout=10)
            assert done["second"].wait(timeout=10)
            assert all(t is not STOP for t in events["first"][:-1])
            assert all(t is not STOP for t in events["second"][:-1])
            # STOP callbacks fire at commit, but the overlap loop may still
            # hold the FINAL submitted step in the executor slot (the step
            # launched past the terminal one) for one more iteration. Wait
            # for the drain instead of racing it.
            deadline = time.time() + 5
            while scheduler._executor.peek_pending() is not None:
                if time.time() > deadline:
                    pytest.fail("overlap loop left a step pending after finish")
                time.sleep(0.01)
            stats = scheduler.get_stats()
            assert stats["in_flight_tasks"] == 0
            assert stats["kv_cache_tasks"] == 0
    finally:
        scheduler.stop()


def test_overlap_loop_matches_synchronous_tokens(device):
    """Overlap decode commits every step: token stream identical to sync.

    Regression gate for the clear_pending bug: the steady overlap branch
    used to detach the JUST-SUBMITTED pending step (the executor slot
    already held the new step, not the one being committed), so every
    other steady iteration's tokens never committed -- requests finished at
    the KV cap with half their tokens and interleaved-token detext.
    Token-count gates cannot see this on small configs (the requests still
    finish); only the full sequence comparison against the synchronous
    run_batch reference catches it.
    """
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = FakeTokenizer()
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
        enable_overlap=True,
    )

    # FakeTokenizer.encode returns the batch shape ([[ids]]); add_request's
    # contract is the flat single-string shape, so unwrap here (same
    # workaround as the overlap e2e test).
    class _UnwrappingTokenizer:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "stop_ids", inner.stop_ids)

        def encode(self, prompt, **kw):
            out = self._inner.encode(prompt, **kw)
            return out[0] if isinstance(out[0], list) else out

        def __getattr__(self, name):
            return getattr(self._inner, name)

    scheduler._requests.tokenizer = _UnwrappingTokenizer(tokenizer)

    # StreamDecoder needs the Rust handle FakeTokenizer lacks; emit the
    # token id itself as the "text" so the sink sees every token.
    class _IdDecoder:
        def __init__(self, tokenizer):
            pass

        def push(self, token_id):
            return str(token_id)

    class _TokenSink(BatchedStreamCallback):
        def __init__(self):
            self.events: list = []

        def __call__(self, batch):
            self.events.extend(batch)

    prompts = ["a" * 8, "b" * 8, "c" * 8]
    sink = _TokenSink()
    scheduler.start()
    try:
        # Synchronous reference: same model, same prompts, greedy.
        reference = scheduler.run_batch(
            [[ord(c) for c in p] for p in prompts], max_tokens=12, temperature=0
        )
        assert all(len(ids) == 12 for ids in reference)

        with patch("astrai.inference.core.request.StreamDecoder", _IdDecoder):
            request_ids = [
                scheduler.add_request(
                    p, max_tokens=12, stream_callback=sink, temperature=0
                )
                for p in prompts
            ]
            deadline = time.time() + 30
            while time.time() < deadline:
                stops = sum(1 for _tid, tok in sink.events if tok is STOP)
                if stops == 3:
                    break
                time.sleep(0.01)

        by_task = {tid: [] for tid in request_ids}
        for tid, token in sink.events:
            if token is not STOP:
                by_task[tid].append(int(token))
        for i, tid in enumerate(request_ids):
            assert by_task[tid] == list(reference[i]), (
                f"request {i} overlap stream {by_task[tid]} != reference "
                f"{list(reference[i])}"
            )
        assert scheduler._executor.peek_pending() is None
    finally:
        scheduler.stop()


def test_stop_leaves_state_intact_when_loop_does_not_drain():
    """stop() must not clear queues under a live loop (KV double-free race).

    A loop stuck longer than the 2s join used to be followed blindly by
    _abort_and_clear + handle reset: the still-running thread would then
    free the same slots again, and a second start() could race the old
    loop.  The fixed contract: on drain failure stop() keeps the handle
    and the request state, and start() refuses to spawn a second loop.
    """
    scheduler = object.__new__(Scheduler)
    scheduler._stop_event = threading.Event()
    scheduler._requests = MagicMock()
    scheduler._loop_thread = MagicMock()
    scheduler._loop_thread.is_alive.return_value = True
    # join "times out": is_alive stays True on both probes.

    scheduler.stop()

    # The live loop's world is left untouched.
    assert scheduler._loop_thread is not None
    scheduler._requests.clear_queues.assert_not_called()
    scheduler._requests.get_running_requests.assert_not_called()

    # start() refuses to launch a second loop alongside the live one.
    with patch("astrai.inference.core.scheduler.threading.Thread") as TH:
        scheduler.start()
        TH.assert_not_called()


def test_stop_clears_state_when_loop_drains_normally():
    """After a clean drain the handle resets and terminal cleanup runs."""
    scheduler = object.__new__(Scheduler)
    scheduler._stop_event = threading.Event()
    scheduler._requests = MagicMock()
    scheduler._requests.get_running_requests.return_value = []
    scheduler._requests.get_waiting_requests.return_value = []
    scheduler._loop_thread = MagicMock()
    scheduler._loop_thread.is_alive.return_value = True  # first probe (before join)

    def join_side_effect(timeout=None):
        scheduler._loop_thread.is_alive.return_value = False

    scheduler._loop_thread.join.side_effect = join_side_effect

    scheduler.stop()

    assert scheduler._loop_thread is None
    scheduler._requests.clear_queues.assert_called_once()


def test_admission_rejects_requests_that_can_never_fit(device):
    """The livelock guard: a prompt larger than the whole paged pool is
    terminated (FINISH_REJECTED) instead of retrying alloc forever."""
    import torch as _torch

    from astrai.inference.core.scheduler import Scheduler
    from astrai.model.transformer import AutoRegressiveLM
    from tests.helpers import FakeTokenizer, make_rollout_config

    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=_torch.bfloat16).eval()
    scheduler = Scheduler(
        model=model,
        tokenizer=FakeTokenizer(),
        max_batch_size=2,
        max_seq_len=64,
        enable_cuda_graph=False,
        page_size=2,
        kv_tokens=16,  # 8 pages × 2 tokens = 16 token slots total
    )
    sink_events = []
    from astrai.inference.core.scheduler import OutputEventSink

    class _Capture(OutputEventSink):
        def __call__(self, events):
            sink_events.extend(events)

    scheduler.set_event_sink(_Capture())
    try:
        scheduler.start()
        # 60-token prompt can never fit a 16-slot pool.
        rid = scheduler.add_request(prompt="x" * 60, max_tokens=4)
        deadline = time.time() + 10
        while time.time() < deadline:
            if any(
                getattr(e, "request_id", None) == rid
                and getattr(e, "finish_reason", None) == "rejected"
                for e in sink_events
            ):
                break
            time.sleep(0.05)
        else:
            pytest.fail("oversized request was never rejected")
        # And the queue drained: no spinning leftovers.
        deadline = time.time() + 5
        while time.time() < deadline and scheduler._requests.get_waiting_requests():
            time.sleep(0.05)
        assert not scheduler._requests.get_waiting_requests()
    finally:
        scheduler.stop()


def test_paged_pool_selected_via_scheduler_kwargs(device):
    """page_size/kv_tokens reach the BlockPool: paged strategy + prefix cache."""
    import torch as _torch

    from astrai.inference.core.scheduler import Scheduler
    from astrai.model.transformer import AutoRegressiveLM
    from tests.helpers import FakeTokenizer, make_rollout_config

    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=_torch.bfloat16).eval()
    scheduler = Scheduler(
        model=model,
        tokenizer=FakeTokenizer(),
        max_batch_size=2,
        max_seq_len=64,
        enable_cuda_graph=False,
        page_size=4,
        kv_tokens=128,
    )
    try:
        assert not scheduler._cache.contiguous
        assert scheduler._cache.page_size == 4
        assert scheduler._cache.n_tokens == 128
        # End-to-end: generation works on the paged path.
        out = scheduler.run_batch([[10, 11, 12, 13, 14]], max_tokens=3, temperature=0.0)
        assert len(out[0]) == 3
    finally:
        scheduler.stop()


def test_chunked_prefill_matches_whole_prompt_greedy_tokens(device):
    """Chunked prefill is token-identical to whole-prompt prefill.

    Same model, same prompts, greedy: a small budget forces each prompt
    through multiple continuation chunks (KV-only forwards) before its
    final chunk samples; the sampled sequence must equal the unchunked
    reference exactly.  Also asserts the budget actually split the work
    (chunked forward count > reference forward count).
    """
    import torch as _torch

    from astrai.inference.core.scheduler import Scheduler
    from astrai.model.transformer import AutoRegressiveLM
    from tests.helpers import FakeTokenizer, make_rollout_config

    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=_torch.bfloat16).eval()

    prompts = [[10, 11, 12, 13, 14, 15, 16, 17], [40, 41, 42, 43], [5, 6, 7]]

    ref_sched = Scheduler(
        model=model,
        tokenizer=FakeTokenizer(),
        max_batch_size=8,
        max_seq_len=64,
        enable_cuda_graph=False,
    )
    fwd_calls = []
    orig_prefill = ref_sched._executor.execute_prefill

    def counting_prefill(requests, **kw):
        fwd_calls.append(
            sum(kw.get("num_tokens") or [len(r.prompt_ids) for r in requests])
        )
        return orig_prefill(requests, **kw)

    ref_sched._executor.execute_prefill = counting_prefill
    try:
        reference = ref_sched.run_batch(
            [list(p) for p in prompts], max_tokens=5, temperature=0.0
        )
    finally:
        ref_sched.stop()
    ref_forwards = len(fwd_calls)

    chunked = Scheduler(
        model=model,
        tokenizer=FakeTokenizer(),
        max_batch_size=8,
        max_seq_len=64,
        enable_cuda_graph=False,
        token_budget=3,  # every prompt needs >=2 windows
    )
    try:
        # Sanity: budget must actually be in force.
        assert chunked._stepper._token_budget == 3
        chunked_result = chunked.run_batch(
            [list(p) for p in prompts], max_tokens=5, temperature=0.0
        )
    finally:
        chunked.stop()

    assert chunked_result == reference, (chunked_result, reference)
    assert all(len(ids) == 5 for ids in chunked_result)


def test_chunked_prefill_budget_caps_forward_tokens(device):
    """No prefill forward under a budget carries more than budget tokens."""
    import torch as _torch

    from astrai.inference.core.scheduler import Scheduler
    from astrai.model.transformer import AutoRegressiveLM
    from tests.helpers import FakeTokenizer, make_rollout_config

    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=_torch.bfloat16).eval()
    scheduler = Scheduler(
        model=model,
        tokenizer=FakeTokenizer(),
        max_batch_size=8,
        max_seq_len=64,
        enable_cuda_graph=False,
        token_budget=4,
    )
    seen = []
    orig = scheduler._executor.execute_prefill

    def spy(requests, **kw):
        seen.append(sum(kw.get("num_tokens") or []))
        return orig(requests, **kw)

    scheduler._executor.execute_prefill = spy
    try:
        out = scheduler.run_batch(
            [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]], max_tokens=2, temperature=0.0
        )
        assert len(out[0]) == 2
    finally:
        scheduler.stop()
    assert seen, "no prefill forwards observed"
    assert max(seen) <= 4, seen
    assert len(seen) >= 3, f"expected multiple chunks, got {seen}"
