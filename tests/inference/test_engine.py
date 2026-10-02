"""Unit tests for GenerateResult accumulator and InferenceEngine.generate()."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from astrai.extension import TorchNativeBackend, attn_backend
from astrai.inference import STOP
from astrai.inference.core.events import (
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.frontend.engine import (
    GenerateResult,
    InferenceEngine,
    build_engine,
)
from astrai.inference.frontend.tracking import RequestTracker
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


class _CharacterDecoder:
    """Deterministic CPU decoder for frontend lifecycle tests."""

    def __init__(self, tokenizer):
        pass

    def push(self, token_id):
        return chr(token_id) if token_id > 2 else ""


@pytest.fixture
def event_engine(monkeypatch):
    monkeypatch.setattr(
        "astrai.inference.frontend.output_processor.StreamDecoder", _CharacterDecoder
    )
    with patch("astrai.inference.frontend.engine.Scheduler"):
        engine = InferenceEngine(MagicMock(), FakeTokenizer(), max_seq_len=32)
    engine._core = MagicMock()
    engine._core.abort_request.side_effect = lambda rid: (
        engine._tracker.sink([RequestFinished(rid, FINISH_CANCELLED)]) or True
    )
    _emit_on_submission(engine)
    return engine


def _emit_on_submission(engine, text="", reason=None, error=False):
    def emit(rid, prompt_ids):
        events = [TokenDelta(rid, ord(char), i + 1) for i, char in enumerate(text)]
        if error:
            events.append(RequestError(rid, "test", "failed"))
        elif reason is not None:
            events.append(RequestFinished(rid, reason, len(prompt_ids), len(text)))
        engine._tracker.sink(events)

    def send_request(**kwargs):
        rid = kwargs["request_id"]
        emit(rid, kwargs["prompt_ids"])
        return rid

    def send_requests(**kwargs):
        for rid, ids in zip(kwargs["request_ids"], kwargs["prompts_ids"]):
            emit(rid, ids)
        return kwargs["request_ids"]

    engine._core.send_request.side_effect = send_request
    engine._core.send_requests.side_effect = send_requests


def _assert_frontend_released(engine):
    tracker = engine._tracker
    assert not tracker._finished
    assert not tracker.sink._queues
    assert not tracker.sink._async_subscribers
    assert not tracker.sink._terminal


def _collect_async(stream):
    async def collect():
        return [chunk async for chunk in stream]

    return asyncio.run(asyncio.wait_for(collect(), timeout=2))


@pytest.mark.parametrize(
    ("text", "stop"), [("ab!ignored", "!"), ("abENDignored", "END")]
)
def test_generate_events_text_stop_aborts_before_yielding_final(
    event_engine, text, stop
):
    engine = event_engine
    _emit_on_submission(engine, text)
    prompt = "p" * 40  # Usage must reflect the actual, left-truncated prompt.

    async def consume():
        stream = engine.generate_events(prompt, stop_sequences=[stop])
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            if chunk.is_final:
                engine._core.abort_request.assert_called_once()
                _assert_frontend_released(engine)
                break
        await stream.aclose()
        return chunks

    chunks = asyncio.run(asyncio.wait_for(consume(), timeout=2))
    final = chunks[-1]
    assert "".join(chunk.text for chunk in chunks) == "ab"
    assert final.is_final and final.stopped
    assert final.stop_sequence == stop
    assert final.finish_reason == "stop"
    assert final.prompt_tokens == 32
    assert final.completion_tokens == len("ab" + stop)
    assert len(final.current_token_ids) == final.completion_tokens
    assert sum(chunk.is_final for chunk in chunks) == 1
    engine._core.abort_request.assert_called_once()


@pytest.mark.parametrize("reason", [FINISH_STOP_TOKEN, FINISH_LENGTH, FINISH_CANCELLED])
def test_generate_events_core_terminal_flushes_tail_without_abort(event_engine, reason):
    _emit_on_submission(event_engine, "hello EN", reason)
    chunks = _collect_async(event_engine.generate_events("hi", stop_sequences=["END"]))
    assert "".join(chunk.text for chunk in chunks) == "hello EN"
    final = chunks[-1]
    assert final.text == "EN"
    assert final.finish_reason == ("stop" if reason == FINISH_STOP_TOKEN else reason)
    assert final.prompt_tokens == 2
    assert final.completion_tokens == len("hello EN")
    assert not final.stopped
    assert sum(chunk.is_final for chunk in chunks) == 1
    event_engine._core.abort_request.assert_not_called()
    _assert_frontend_released(event_engine)


def test_generate_events_stop_after_core_terminal_needs_no_abort(event_engine):
    _emit_on_submission(event_engine, "ab!ignored", FINISH_LENGTH)
    chunks = _collect_async(event_engine.generate_events("hi", stop_sequences=["!"]))
    assert chunks[-1].stopped
    assert "".join(chunk.text for chunk in chunks) == "ab"
    assert chunks[-1].prompt_tokens == 2
    event_engine._core.abort_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
@pytest.mark.parametrize("exit_kind", ["close", "exception", "cancel"])
def test_async_consumer_exit_cancels_and_releases(event_engine, method, exit_kind):
    _emit_on_submission(event_engine, "a")

    async def consume():
        stream = getattr(event_engine, method)("hi")
        await anext(stream)
        if exit_kind == "close":
            await stream.aclose()
        elif exit_kind == "exception":
            with pytest.raises(RuntimeError, match="consumer failed"):
                await stream.athrow(RuntimeError("consumer failed"))
        else:
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await stream.aclose()

    asyncio.run(asyncio.wait_for(consume(), timeout=2))
    event_engine._core.abort_request.assert_called_once()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
def test_async_close_after_core_terminal_is_not_cancelled(event_engine, method):
    _emit_on_submission(event_engine, "abc", FINISH_LENGTH)

    async def consume():
        stream = getattr(event_engine, method)("hi")
        await anext(stream)  # Terminal is queued, but not folded yet.
        await stream.aclose()

    asyncio.run(asyncio.wait_for(consume(), timeout=2))
    event_engine._core.abort_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
def test_unstarted_async_stream_does_not_submit_or_leak(event_engine, method):
    stream = getattr(event_engine, method)("hi")
    asyncio.run(stream.aclose())
    event_engine._core.send_request.assert_not_called()
    event_engine._core.abort_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
def test_async_stream_captures_backend_before_lazy_submission(event_engine, method):
    _emit_on_submission(event_engine, reason=FINISH_LENGTH)
    with attn_backend("torch_native"):
        stream = getattr(event_engine, method)("hi")
    _collect_async(stream)
    assert isinstance(
        event_engine._core.send_request.call_args.kwargs["backend"], TorchNativeBackend
    )


def test_unstarted_sync_stream_does_not_submit_or_leak(event_engine):
    stream = event_engine.generate("hi", stream=True)
    stream.close()
    event_engine._core.send_requests.assert_not_called()
    event_engine._core.abort_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("core_finished", [False, True])
def test_sync_stream_close_cancels_only_unfinished_core(event_engine, core_finished):
    _emit_on_submission(event_engine, "abc", FINISH_LENGTH if core_finished else None)
    stream = event_engine.generate("hi", stream=True)
    assert next(stream) == "a"
    stream.close()
    assert event_engine._core.abort_request.call_count == int(not core_finished)
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize(
    "method", ["generate", "stream", "generate_events", "generate_async"]
)
def test_request_error_terminates_every_frontend_path(
    event_engine, monkeypatch, method
):
    _emit_on_submission(event_engine, "ab", error=True)
    original_wait = event_engine._tracker.wait

    def wait(rid, timeout=None):
        assert event_engine._tracker.is_finished(rid), "unexpected wait for terminal"
        return original_wait(rid, timeout=timeout)

    monkeypatch.setattr(event_engine._tracker, "wait", wait)
    if method == "generate":
        assert event_engine.generate("hi") == "ab"
    elif method == "stream":
        # A fold that fails to mark RequestError terminal must fail, not hang.
        monkeypatch.setattr(
            event_engine._tracker,
            "wait",
            MagicMock(side_effect=AssertionError("stream hung")),
        )
        assert "".join(event_engine.generate("hi", stream=True)) == "ab"
    else:
        chunks = _collect_async(getattr(event_engine, method)("hi"))
        if method == "generate_events":
            assert "".join(chunk.text for chunk in chunks) == "ab"
            assert chunks[-1].finish_reason == "aborted"
            assert chunks[-1].prompt_tokens == 2
            assert chunks[-1].completion_tokens == 2
        else:
            assert "".join(chunks) == "ab"
    event_engine._core.abort_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("max_tokens", [0, -1, None])
@pytest.mark.parametrize(
    "method", ["generate", "stream", "generate_events", "generate_async"]
)
def test_zero_token_terminal_does_not_wait_for_token_delta(
    event_engine, monkeypatch, max_tokens, method
):
    _emit_on_submission(event_engine, reason=FINISH_LENGTH)
    prompt = "p" * 32 if max_tokens is None else "hi"
    if method in ("generate", "stream"):
        if method == "stream":
            monkeypatch.setattr(
                event_engine._tracker,
                "wait",
                MagicMock(side_effect=AssertionError("stream hung")),
            )
        result = event_engine.generate(
            prompt, stream=method == "stream", max_tokens=max_tokens
        )
        assert (list(result) if method == "stream" else result) == (
            [] if method == "stream" else ""
        )
        if max_tokens is not None:
            event_engine._core.send_requests.assert_not_called()
    else:
        chunks = _collect_async(
            getattr(event_engine, method)(prompt, max_tokens=max_tokens)
        )
        if method == "generate_events":
            assert len(chunks) == 1
            assert chunks[0].text == ""
            assert chunks[0].finish_reason == "length"
            assert chunks[0].prompt_tokens == len(prompt)
            assert chunks[0].completion_tokens == 0
        else:
            assert chunks == []
    event_engine._core.abort_request.assert_not_called()
    _assert_frontend_released(event_engine)


def test_non_streaming_still_bulk_decodes_once_without_helper_threads(
    event_engine, monkeypatch
):
    _emit_on_submission(event_engine, "abc", FINISH_LENGTH)
    event_engine.tokenizer._tokenizer = object()
    event_engine.tokenizer.decode = MagicMock(wraps=event_engine.tokenizer.decode)
    monkeypatch.setattr(
        "astrai.inference.frontend.engine.OutputProcessor",
        MagicMock(side_effect=AssertionError("incremental fold used")),
    )
    monkeypatch.setattr(
        threading,
        "Thread",
        MagicMock(side_effect=AssertionError("helper thread started")),
    )
    assert event_engine.generate(["hi", "there"]) == ["abc", "abc"]
    assert event_engine.tokenizer.decode.call_count == 2
    event_engine._core.abort_request.assert_not_called()
    _assert_frontend_released(event_engine)


def test_blocking_timeout_cancels_only_unfinished_requests(event_engine, monkeypatch):
    def submit(**kwargs):
        event_engine._tracker.sink(
            [RequestFinished(kwargs["request_ids"][0], FINISH_LENGTH, 2, 0)]
        )
        return kwargs["request_ids"]

    event_engine._core.send_requests.side_effect = submit
    monkeypatch.setattr("astrai.inference.frontend.engine._GENERATE_TIMEOUT_S", 0.01)
    with pytest.raises(TimeoutError, match="1/2 completed"):
        event_engine.generate(["hi", "there"])
    request_ids = event_engine._core.send_requests.call_args.kwargs["request_ids"]
    event_engine._core.abort_request.assert_called_once_with(request_ids[1])
    _assert_frontend_released(event_engine)


def test_blocking_wait_exception_cancels_and_releases_batch(event_engine, monkeypatch):
    monkeypatch.setattr(
        event_engine._tracker, "wait", MagicMock(side_effect=KeyboardInterrupt)
    )
    with pytest.raises(KeyboardInterrupt):
        event_engine.generate(["hi", "there"])
    assert event_engine._core.abort_request.call_count == 2
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize(
    "method", ["generate", "stream", "generate_events", "generate_async"]
)
def test_submission_failure_releases_frontend_state(event_engine, method):
    event_engine._core.send_request.side_effect = RuntimeError("submit failed")
    event_engine._core.send_requests.side_effect = RuntimeError("submit failed")
    with pytest.raises(RuntimeError, match="submit failed"):
        if method == "generate":
            event_engine.generate("hi")
        elif method == "stream":
            list(event_engine.generate("hi", stream=True))
        else:
            _collect_async(getattr(event_engine, method)("hi"))
    event_engine._core.abort_request.assert_called_once()
    _assert_frontend_released(event_engine)


def test_output_fold_exception_cancels_and_releases_stream(event_engine, monkeypatch):
    _emit_on_submission(event_engine, "a")
    monkeypatch.setattr(
        "astrai.inference.frontend.engine.OutputProcessor.push",
        MagicMock(side_effect=RuntimeError("fold failed")),
    )
    with pytest.raises(RuntimeError, match="fold failed"):
        _collect_async(event_engine.generate_events("hi"))
    event_engine._core.abort_request.assert_called_once()
    _assert_frontend_released(event_engine)


def test_async_backlog_is_bounded_for_full_context_without_token_loss(event_engine):
    event_engine._max_seq_len = 5000
    _emit_on_submission(event_engine, "a" * 4500, FINISH_LENGTH)
    chunks = _collect_async(event_engine.generate_events("hi"))
    assert "".join(chunk.text for chunk in chunks) == "a" * 4500
    assert chunks[-1].completion_tokens == 4500
    event_engine._core.abort_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
def test_cancellation_before_first_token_releases_request(event_engine, method):
    async def consume():
        stream = getattr(event_engine, method)("hi")
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        event_engine._core.send_request.assert_called_once()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await stream.aclose()

    asyncio.run(asyncio.wait_for(consume(), timeout=2))
    event_engine._core.abort_request.assert_called_once()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize(
    "method", ["generate", "stream", "generate_events", "generate_async"]
)
@pytest.mark.parametrize("full_prompt", [False, True])
def test_real_cpu_engine_zero_output_boundaries(monkeypatch, method, full_prompt):
    model, _ = make_model("cpu", max_position_embeddings=16)
    monkeypatch.setattr("astrai.inference.frontend.engine._GENERATE_TIMEOUT_S", 2)
    with build_engine(
        model=model,
        tokenizer=FakeTokenizer(),
        device=None,
        dtype=None,
        max_seq_len=16,
        max_batch_size=1,
        enable_cuda_graph=False,
        backend="torch_native",
    ) as engine:
        original_wait = engine._tracker.wait
        waits = 0

        def bounded_wait(rid, timeout=None):
            nonlocal waits
            waits += 1
            assert waits < 40, "zero-output stream waited without a terminal"
            return original_wait(rid, timeout=timeout)

        monkeypatch.setattr(engine._tracker, "wait", bounded_wait)
        engine._core.abort_request = MagicMock(wraps=engine._core.abort_request)
        prompt, max_tokens = ("p" * 32, None) if full_prompt else ("hi", 0)
        if method == "generate":
            assert engine.generate(prompt, max_tokens=max_tokens) == ""
        elif method == "stream":
            assert (
                list(engine.generate(prompt, stream=True, max_tokens=max_tokens)) == []
            )
        else:
            chunks = _collect_async(
                getattr(engine, method)(prompt, max_tokens=max_tokens)
            )
            if method == "generate_events":
                assert len(chunks) == 1
                assert chunks[0].text == ""
                assert chunks[0].finish_reason == "length"
                assert chunks[0].prompt_tokens == (16 if full_prompt else 2)
                assert chunks[0].completion_tokens == 0
            else:
                assert chunks == []
        engine._core.abort_request.assert_not_called()
        _assert_frontend_released(engine)


def test_tracker_accepts_one_terminal_and_reclaims_all_state():
    tracker = RequestTracker()
    done = tracker.register("r", maxlen=4)
    terminal_callback = MagicMock(wraps=tracker.mark_finished)
    tracker.sink._on_terminal = terminal_callback
    terminal = RequestFinished("r", FINISH_CANCELLED, 2, 1)
    tracker.sink([TokenDelta("r", 11, 1), terminal, terminal, TokenDelta("r", 12, 2)])
    tracker.sink([RequestError("r", "late", "ignored")])
    assert done.is_set()
    assert tracker.drain("r") == [TokenDelta("r", 11, 1), terminal]
    terminal_callback.assert_called_once_with("r")
    assert tracker.sink._queues["r"].maxlen == 4
    tracker.unregister("r")
    tracker.sink([terminal])
    assert not tracker._finished
    assert not tracker.sink._queues
    assert not tracker.sink._terminal
    terminal_callback.assert_called_once()
    replacement = tracker.register("r")
    assert not replacement.is_set()
    tracker.sink([terminal])
    assert replacement.is_set()
    assert tracker.drain("r") == [terminal]
    tracker.unregister("r")


def test_tracker_publishes_core_terminal_before_async_delivery():
    tracker = RequestTracker()
    tracker.register("r")
    loop = MagicMock()
    queue, pending = tracker.subscribe_async("r", loop)
    assert pending == []

    def deliver(callback, deliveries):
        assert tracker.is_finished("r")
        callback(deliveries)

    loop.call_soon_threadsafe.side_effect = deliver
    terminal = RequestFinished("r", FINISH_LENGTH)
    tracker.sink([terminal])
    assert queue.get_nowait() == [terminal]
    tracker.unregister("r")


def test_tracker_closed_loop_preserves_interleaved_terminal_backlog():
    tracker = RequestTracker()
    loop = MagicMock()
    loop.call_soon_threadsafe.side_effect = RuntimeError("loop closed")
    for rid in ("a", "b"):
        tracker.register(rid)
        tracker.subscribe_async(rid, loop)
    tracker.sink(
        [
            TokenDelta("a", 11, 1),
            TokenDelta("b", 12, 1),
            RequestFinished("a", FINISH_LENGTH, 2, 1),
            RequestFinished("b", FINISH_CANCELLED, 2, 1),
        ]
    )
    for rid, token, reason in (("a", 11, FINISH_LENGTH), ("b", 12, FINISH_CANCELLED)):
        assert tracker.is_finished(rid)
        assert tracker.drain(rid) == [
            TokenDelta(rid, token, 1),
            RequestFinished(rid, reason, 2, 1),
        ]
        tracker.unregister(rid)
    assert not tracker.sink._async_subscribers
    assert not tracker.sink._terminal
