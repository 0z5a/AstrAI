"""Unit tests for Request and RequestManager."""

from unittest.mock import MagicMock

import pytest

from astrai.inference import (
    STOP,
    BatchedStreamCallback,
    Request,
    RequestManager,
    RequestStatus,
)


class RecordingSink(BatchedStreamCallback):
    """Batch-aware callback capturing every dispatch as one batch."""

    def __init__(self):
        self.batches = []

    def __call__(self, events):
        self.batches.append(events)


def _make_mock_tokenizer():
    t = MagicMock()
    t.encode.return_value = [1, 2, 3, 4, 5]
    t.stop_ids = [0]
    return t


def test_task_default_status_is_pending():
    request = Request("id1", [1, 2, 3])
    assert request.status == RequestStatus.PENDING


def test_task_next_pos():
    request = Request("id1", [1, 2, 3])
    request.input_tokens = 5
    request.mark_prefill_complete()
    assert request.next_pos == 5
    request.advance_kv()
    assert request.next_pos == 6
    request.advance_kv()
    assert request.next_pos == 7


def test_task_is_finished_max_tokens():
    request = Request("id1", [1, 2, 3], max_tokens=2)
    request.output_tokens = 2
    assert request.is_finished([])


def test_task_is_finished_stop_id():
    request = Request("id1", [1, 2, 3])
    request.output_ids = [5, 0]
    assert request.is_finished([0])


def test_task_is_finished_not_yet():
    request = Request("id1", [1, 2, 3], max_tokens=10)
    request.output_ids = [1, 2]
    assert not request.is_finished([0])


def test_task_manager_add_task():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tid = tm.add_request("hello")
    assert tid.startswith("req_")
    assert tm._total_requests == 1
    assert len(tm.waiting) == 1


def test_task_manager_long_prompt_truncated_not_stopped():
    t = _make_mock_tokenizer()
    t.encode.return_value = list(range(9000))
    cb_calls = []

    tm = RequestManager(tokenizer=t, max_seq_len=16)
    tm.add_request("long", stream_callback=lambda tok: cb_calls.append(tok))
    assert len(cb_calls) == 0
    assert len(tm.waiting) == 1
    assert len(tm.waiting[0].prompt_ids) == 16


def test_task_manager_remove_request():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tid = tm.add_request("test")
    tm.cancel_request(tid)
    assert len(tm.waiting) == 0


def test_task_manager_cancel_active_task_defers_removal():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tid = tm.add_request("test")
    requests = tm.pull_waiting(1)
    tm.activate(requests[0])
    assert len(tm.running) == 1
    immediate, cancelled = tm.cancel_request(tid)
    assert cancelled is True
    assert immediate == []
    assert tm.running[0].status == RequestStatus.ABORTED

    removed = tm.remove_finished_requests([])
    assert removed == requests
    assert len(tm.running) == 0
    assert tm.get_stats()["cancelled_total"] == 1


def test_task_manager_pull_candidates_fifo():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("a")
    tm.add_request("b")
    tm.add_request("c")
    pulled = tm.pull_waiting(2)
    assert len(pulled) == 2
    assert pulled[0].prompt_ids == [1, 2, 3, 4, 5]
    assert len(tm.waiting) == 1


def test_task_manager_activate():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("test")
    request = tm.pull_waiting(1)[0]
    tm.activate(request)
    assert request.status == RequestStatus.RUNNING
    assert request in tm.running


def test_task_manager_return_to_waiting():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("a")
    tm.add_request("b")
    t1 = tm.pull_waiting(1)[0]
    tm.return_to_waiting([t1])
    assert len(tm.waiting) == 2
    assert tm.waiting[0] == t1


def test_task_manager_remove_finished_aborted():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("test")
    request = tm.pull_waiting(1)[0]
    tm.activate(request)
    request.status = RequestStatus.ABORTED
    finished = tm.remove_finished_requests([0])
    assert len(finished) == 1
    assert len(tm.running) == 0


def test_task_manager_remove_finished_stop_id():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("test")
    request = tm.pull_waiting(1)[0]
    tm.activate(request)
    request.output_ids = [0]
    request.output_tokens = 1
    finished = tm.remove_finished_requests([0])
    assert len(finished) == 1
    assert request.status == RequestStatus.FINISHED
    assert len(tm.running) == 0


def test_task_manager_has_work():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    assert not tm.has_requests()
    tm.add_request("test")
    assert tm.has_requests()


def test_task_manager_wake():
    import threading

    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    called = threading.Event()

    def waiter():
        tm.wait_for_requests(timeout=5.0)
        called.set()

    t = threading.Thread(target=waiter)
    t.start()
    import time

    time.sleep(0.05)
    tm.wake()
    t.join(timeout=2.0)
    assert called.is_set()


def test_task_manager_get_stats():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("test")
    stats = tm.get_stats()
    assert stats["total_tasks"] == 1
    assert stats["waiting"] == 1
    assert stats["running"] == 0


def test_task_manager_add_task_rejects_empty_prompt():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.tokenizer.encode.return_value = []

    with pytest.raises(ValueError, match="zero tokens"):
        tm.add_request("")


def test_task_manager_cancel_delivers_stop_callback():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    received = []
    tm.add_request("test", stream_callback=received.append)

    immediate, cancelled = tm.cancel_request("does-not-exist")
    assert not cancelled and immediate == [] and received == []

    request_id = next(iter(tm._requests))
    immediate, cancelled = tm.cancel_request(request_id)
    assert cancelled
    assert len(immediate) == 1
    assert received == [STOP]


def test_task_manager_cancel_active_task_delivers_stop_callback():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    received = []
    request_id = tm.add_request("test", stream_callback=received.append)
    request = tm._requests[request_id]
    tm.waiting.clear()
    tm.running.append(request)
    request.status = RequestStatus.RUNNING

    immediate, cancelled = tm.cancel_request(request_id)
    assert cancelled and immediate == []
    assert received == [STOP]


def test_invoke_callbacks_batches_sink_events_and_keeps_plain_per_token():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    plain = []
    tid_plain = tm.add_request("plain", stream_callback=plain.append)
    sink = RecordingSink()
    tid_a = tm.add_request("sink a", stream_callback=sink)
    tid_b = tm.add_request("sink b", stream_callback=sink)

    tm.invoke_callbacks(
        [
            (tid_a, "x"),
            (tid_plain, "p"),
            (tid_b, "y"),
            ("unknown-request", "dropped"),
            (tid_a, STOP),
        ]
    )

    assert plain == ["p"]
    assert sink.batches == [[(tid_a, "x"), (tid_b, "y"), (tid_a, STOP)]]


def test_invoke_callbacks_delivers_single_event_to_batched_sink():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    sink = RecordingSink()
    request_id = tm.add_request("test", stream_callback=sink)

    tm.invoke_callbacks([(request_id, STOP)])

    assert sink.batches == [[(request_id, STOP)]]


def test_cancel_delivers_batched_stop_to_sink():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    sink = RecordingSink()
    request_id = tm.add_request("test", stream_callback=sink)

    immediate, cancelled = tm.cancel_request(request_id)

    assert cancelled
    assert sink.batches == [[(request_id, STOP)]]
