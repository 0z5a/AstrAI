"""Unit tests for GenerateResult accumulator and InferenceEngine.generate()."""

import asyncio
import itertools
import threading
from unittest.mock import MagicMock, patch

import pytest

from astrai.extension import TorchNativeBackend, attn_backend
from astrai.inference import STOP
from astrai.inference.frontend.engine import (
    GenerateResult,
    InferenceEngine,
    build_engine,
)
from tests.helpers import FakeTokenizer, make_model


def _make_engine_mocks(decode=None):
    """Build the standard mock model/tokenizer pair used by engine tests."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.stop_ids = [0]
    if decode is not None:
        mock_tokenizer.decode.return_value = decode
    return mock_model, mock_tokenizer


def test_result_append_multiple_tasks():
    r = GenerateResult(count=3)
    r.append("a", 0)
    r.append("b", 1)
    r.append("c", 2)
    assert r.results[0] == "a"
    assert r.results[1] == "b"
    assert r.results[2] == "c"


def test_result_stop_marks_complete():
    r = GenerateResult(count=2)
    r.append("text", 0)
    r.append(STOP, 0)
    r.append("more", 1)
    assert r._done[0] is True
    assert r._done[1] is False
    assert r._completed == 1


def test_result_stop_does_not_double_count():
    r = GenerateResult(count=1)
    r.append(STOP, 0)
    r.append(STOP, 0)
    assert r._completed == 1


def test_result_append_batch_updates_state_in_one_commit():
    r = GenerateResult(count=2)
    r.append_batch([(0, "he"), (1, "wo"), (0, "llo"), (1, "rld")])
    r.append_batch([(0, STOP), (1, STOP)])
    assert r.results == ["hello", "world"]
    assert r._completed == 2
    assert r.pop_all() == [
        (0, "he"),
        (1, "wo"),
        (0, "llo"),
        (1, "rld"),
        (0, STOP),
        (1, STOP),
    ]


def test_request_tracker_events_only_reach_registered_requests():
    """The frontend mints ids before submission, so events for unregistered
    ids are dropped by the sink (they cannot exist on the happy path);
    once registered, events land in the per-request queue in order."""
    from astrai.inference.core.events import (
        FINISH_LENGTH,
        RequestFinished,
        TokenDelta,
    )
    from astrai.inference.frontend.tracking import RequestTracker

    tracker = RequestTracker()
    done = tracker.register("t0")
    tracker.sink([TokenDelta("t0", 11, 1), TokenDelta("t0", 12, 2)])
    events = tracker.drain("t0")
    assert [e.token_id for e in events] == [11, 12]
    tracker.sink([TokenFinished := RequestFinished("t0", FINISH_LENGTH, 3, 2)])
    assert tracker.drain("t0") == [TokenFinished]
    tracker.mark_finished("t0")
    assert done.is_set()
    tracker.unregister("t0")


def test_result_pop_all_returns_and_clears():
    r = GenerateResult(count=2)
    r.append("a", 0)
    r.append("b", 1)
    out = r.pop_all()
    assert len(out) == 2
    assert out[0] == (0, "a")
    assert out[1] == (1, "b")
    assert r.pop_all() == []


def test_result_wait_blocks_until_data():
    r = GenerateResult(count=1)

    def delayed_append():
        import time

        time.sleep(0.05)
        r.append("delayed", 0)

    t = threading.Thread(target=delayed_append)
    t.start()
    ok = r.wait(timeout=5.0)
    t.join()
    assert ok
    assert r.results[0] == "delayed"


def test_result_wait_timeout():
    r = GenerateResult(count=1)
    ok = r.wait(timeout=0.01)
    assert not ok


def test_result_wait_completion_non_streaming():
    r = GenerateResult(count=2)

    def finish_later():
        import time

        time.sleep(0.05)
        r.append(STOP, 0)
        time.sleep(0.05)
        r.append(STOP, 1)

    t = threading.Thread(target=finish_later)
    t.start()
    r.wait_completion()
    t.join()
    assert r._completed == 2


def test_result_get_results():
    r = GenerateResult(count=2)
    r.append("hello", 0)
    r.append("world", 1)
    results = r.get_results()
    assert results == ["hello", "world"]


def _drive_events(engine, events_by_id):
    """Push output events through the engine's installed sink (mock schedulers
    emit nothing on their own; tests drive the event protocol directly)."""
    sink = engine.scheduler._event_sink
    flat = []
    for rid, events in events_by_id.items():
        flat.extend(events)
    sink(flat)


def _mock_add_requests(events_by_id, tokenizer_ids=None):
    """Build an add_requests side effect that returns ids and, on the NEXT
    scheduler tick (i.e. when the test drives the sink), replays events."""

    def fake(prompts, **kw):
        # Return core-minted ids matching the ids the engine registered.
        return list(events_by_id.keys())[: len(prompts)]

    return fake


def test_engine_generate_non_streaming_single():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="response")
    # Mock tokenizer returns a flat list for both single and batch shapes.
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_requests.side_effect = lambda prompts, **kw: [
            f"request-{i}" for i in range(len(prompts))
        ]
        instance.add_request.side_effect = lambda prompt, **kw: "request-0"
        instance.remove_request.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)

        # Drive generation from another thread: emit events after submit.
        from astrai.inference.core.events import (
            FINISH_LENGTH,
            RequestFinished,
            TokenDelta,
        )

        def run():
            result = eng.generate("hello")
            return result

        # The events must arrive after generate() registered the request but
        # the mock scheduler never runs a loop, so a helper thread polls the
        # tracker and injects the terminal sequence.
        import time

        def inject():
            deadline = time.time() + 5
            while time.time() < deadline:
                if eng._tracker._sink._queues:
                    rid = next(iter(eng._tracker._sink._queues))
                    eng._tracker.sink(
                        [
                            TokenDelta(rid, 11, 1),
                            TokenDelta(rid, 12, 2),
                            RequestFinished(
                                rid, FINISH_LENGTH, prompt_tokens=3, completion_tokens=2
                            ),
                        ]
                    )
                    return
                time.sleep(0.01)

        t = threading.Thread(target=inject, daemon=True)
        t.start()
        result = run()
        t.join(timeout=5)
        assert result != ""


def test_engine_generate_streaming_yields_token_ids():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="tok")
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_requests.side_effect = lambda prompts, **kw: [
            f"request-{i}" for i in range(len(prompts))
        ]
        instance.cancel_request.return_value = True

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        gen = eng.generate("hello", stream=True)

        from astrai.inference.core.events import (
            FINISH_LENGTH,
            RequestFinished,
            TokenDelta,
        )

        def inject():
            import time

            deadline = time.time() + 5
            while time.time() < deadline:
                if eng._tracker._sink._queues:
                    rid = next(iter(eng._tracker._sink._queues))
                    eng._tracker.sink(
                        [
                            TokenDelta(rid, 11, 1),
                            TokenDelta(rid, 12, 2),
                            RequestFinished(rid, FINISH_LENGTH, 3, 2),
                        ]
                    )
                    return
                time.sleep(0.01)

        t = threading.Thread(target=inject, daemon=True)
        t.start()
        tokens = list(gen)
        t.join(timeout=5)
        assert len(tokens) >= 1  # decoder is a mock; at least one fragment


def test_engine_generate_non_streaming_batch():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="r")
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_requests.side_effect = lambda prompts, **kw: [
            f"request-{i}" for i in range(len(prompts))
        ]
        instance.remove_request.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=2)

        from astrai.inference.core.events import (
            FINISH_LENGTH,
            RequestFinished,
            TokenDelta,
        )

        def inject():
            import time

            deadline = time.time() + 5
            while time.time() < deadline:
                queues = eng._tracker._sink._queues
                if len(queues) >= 2:
                    ids = list(queues)
                    events = []
                    for rid in ids:
                        events.extend(
                            [
                                TokenDelta(rid, 11, 1),
                                RequestFinished(rid, FINISH_LENGTH, 3, 1),
                            ]
                        )
                    eng._tracker.sink(events)
                    return
                time.sleep(0.01)

        t = threading.Thread(target=inject, daemon=True)
        t.start()
        results = eng.generate(["hello", "world"])
        t.join(timeout=5)
        assert len(results) == 2


def test_engine_generate_zero_max_tokens_returns_empty():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.remove_request.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=2)
        assert eng.generate(["hello", "world"], max_tokens=0) == ["", ""]
        instance.add_requests.assert_not_called()


def test_engine_generate_zero_max_tokens_stream_is_empty():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        assert list(eng.generate("hello", stream=True, max_tokens=0)) == []
        instance.add_requests.assert_not_called()


def test_engine_exposes_release_resume_lifecycle():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        scheduler = MockSched.return_value
        scheduler.runtime_released = False
        scheduler.release.return_value = True
        scheduler.resume.return_value = True
        engine = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)

        assert engine.runtime_released is False
        assert engine.release() is True
        assert engine.resume() is True
        scheduler.release.assert_called_once_with()
        scheduler.resume.assert_called_once_with()


def test_engine_passes_backend_to_scheduler():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        InferenceEngine(
            mock_model,
            mock_tokenizer,
            max_batch_size=1,
            backend="torch_native",
        )

    assert MockSched.call_args.kwargs["backend"] == "torch_native"


@pytest.mark.parametrize("enable_overlap", [False, True])
def test_engine_passes_overlap_to_scheduler(enable_overlap):
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        InferenceEngine(
            mock_model,
            mock_tokenizer,
            max_batch_size=1,
            enable_overlap=enable_overlap,
        )

    assert MockSched.call_args.kwargs["enable_overlap"] is enable_overlap


def test_generate_captures_calling_backend_context():
    mock_model, mock_tokenizer = _make_engine_mocks()
    captured = []

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.cancel_request.return_value = True

        def fake_add_tasks(prompts, **kwargs):
            captured.append(kwargs["backend"])
            return [f"request-{i}" for i in range(len(prompts))]

        instance.add_requests.side_effect = fake_add_tasks
        engine = InferenceEngine(mock_model, mock_tokenizer)
        with attn_backend("torch_native"):
            gen = engine.generate("hello", stream=True)
            # Terminate via the event protocol (the mock core has no loop).
            from astrai.inference.core.events import FINISH_LENGTH, RequestFinished

            def inject():
                import time

                for _ in range(500):
                    qs = engine._tracker._sink._queues
                    if qs:
                        rid = next(iter(qs))
                        engine._tracker.sink(
                            [RequestFinished(rid, FINISH_LENGTH, 1, 0)]
                        )
                        return
                    time.sleep(0.01)

            t = threading.Thread(target=inject, daemon=True)
            t.start()
            assert list(gen) == []

    assert len(captured) == 1
    assert isinstance(captured[0], TorchNativeBackend)


def test_build_engine_from_live_objects_starts_scheduler():
    model, _ = make_model("cpu", max_position_embeddings=64)
    tokenizer = FakeTokenizer()
    engine = build_engine(
        model=model,
        tokenizer=tokenizer,
        device=None,
        dtype=None,
        max_batch_size=2,
    )
    try:
        assert isinstance(engine, InferenceEngine)
        assert engine.tokenizer is tokenizer
        assert engine.scheduler._stop_event.is_set() is False
    finally:
        engine.shutdown()


def test_build_engine_passes_engine_kwargs_through():
    model, _ = make_model("cpu", max_position_embeddings=64)
    backend = TorchNativeBackend()
    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.cancel_request.return_value = True

        def fake_add_tasks(*args, **k):
            return [
                f"request-{i}"
                for i in range(
                    len(args[0]) if args else (len(k.get("prompts", [])) or 1)
                )
            ]

        instance.add_requests.side_effect = fake_add_tasks
        engine = build_engine(
            model=model,
            tokenizer=FakeTokenizer(),
            device=None,
            dtype=None,
            cache=object(),
            enable_cuda_graph=False,
            backend=backend,
            enable_overlap=True,
        )
        gen = engine.generate("hi", stream=True)

        from astrai.inference.core.events import FINISH_LENGTH, RequestFinished

        def inject():
            import time

            for _ in range(500):
                qs = engine._tracker._sink._queues
                if qs:
                    rid = next(iter(qs))
                    engine._tracker.sink([RequestFinished(rid, FINISH_LENGTH, 1, 0)])
                    return
                time.sleep(0.01)

        t = threading.Thread(target=inject, daemon=True)
        t.start()
        assert list(gen) == []

    kwargs = MockSched.call_args.kwargs
    assert kwargs["cache"] is not None
    assert kwargs["enable_cuda_graph"] is False
    assert kwargs["backend"] is backend
    assert kwargs["enable_overlap"] is True


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"param_path": "x", "model": object()},
            ValueError,
            "not both",
        ),
        ({}, ValueError, "requires param_path"),
        ({"param_path": "/nonexistent-dir-xyz"}, FileNotFoundError, "not found"),
    ],
)
def test_build_engine_rejects_invalid_arguments(kwargs, error, message):
    with pytest.raises(error, match=message):
        build_engine(**kwargs)
