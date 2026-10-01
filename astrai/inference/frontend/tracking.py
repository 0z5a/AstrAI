"""Frontend request tracking: event queues, stream chunks, finish mapping.

Split out of ``engine.py`` so the facade stays a facade: the tracker owns
the bounded per-request queues the scheduler loop appends into, and the
chunk type carries structured output to protocol adapters.
"""

import threading
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from astrai.inference.core.events import (
    RequestError,
    RequestFinished,
)
from astrai.inference.core.scheduler import OutputEventSink


class EventQueueSink(OutputEventSink):
    """Bounded per-request event queues, fed by the scheduler loop thread.

    The loop thread only appends and notifies — consumer code (detokenize,
    protocol formatting, user callbacks) runs wherever the queue is
    drained.  The bound is per request, not global: a slow consumer
    applies backpressure to its own requests only.

    When a terminal event (:class:`RequestFinished` / :class:`RequestError`)
    is queued, ``on_terminal`` fires for that request — the core's event
    contract guarantees exactly one terminal event, queued after all of the
    request's tokens, so the flag doubles as a completion signal for
    consumers that only want the final result.
    """

    def __init__(
        self,
        maxlen: int = 4096,
        on_terminal: Optional[Callable[[str], None]] = None,
    ):
        self._lock = threading.Lock()
        self._queues: Dict[str, Deque[Any]] = {}
        self._maxlen = maxlen
        self._on_terminal = on_terminal

    def register(self, request_id: str, maxlen: Optional[int] = None) -> None:
        with self._lock:
            bound = self._maxlen if maxlen is None else maxlen
            self._queues[request_id] = deque(maxlen=bound)

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._queues.pop(request_id, None)

    def __call__(self, events: List[Any]) -> None:
        # Fast path: bucket events per request under one lock, then notify.
        terminal: List[str] = []
        with self._lock:
            for event in events:
                rid = event.request_id
                queue = self._queues.get(rid)
                if queue is not None:
                    queue.append(event)
                    if isinstance(event, (RequestFinished, RequestError)):
                        terminal.append(rid)
        # Signalled outside the lock, after the terminal event is queued:
        # a woken consumer always finds it in the drain.
        if self._on_terminal is not None:
            for rid in terminal:
                self._on_terminal(rid)


class RequestTracker:
    """Frontend bookkeeping: request_id -> queue + lifecycle flag.

    Mirrors vLLM's ``RequestTracker``: the engine mints the id, registers
    the queue BEFORE the request reaches the scheduler, and therefore no
    event can ever precede its consumer (the old ``_ResultSink`` replay
    buffer existed only because ids were minted in the core).
    """

    def __init__(self):
        # Terminal events flip the finished flag as they are queued (sink
        # callback), so ``wait`` / ``is_finished`` are true the moment the
        # scheduler has finished a request — independent of any consumer.
        self._sink = EventQueueSink(on_terminal=self.mark_finished)
        self._finished: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    @property
    def sink(self) -> EventQueueSink:
        return self._sink

    def register(
        self, request_id: str, maxlen: Optional[int] = None
    ) -> threading.Event:
        self._sink.register(request_id, maxlen=maxlen)
        with self._lock:
            done = self._finished[request_id] = threading.Event()
        return done

    def unregister(self, request_id: str) -> None:
        self._sink.unregister(request_id)
        with self._lock:
            self._finished.pop(request_id, None)

    def is_finished(self, request_id: str) -> bool:
        with self._lock:
            done = self._finished.get(request_id)
        return done is not None and done.is_set()

    def mark_finished(self, request_id: str) -> None:
        with self._lock:
            done = self._finished.get(request_id)
        if done is not None:
            done.set()

    def drain(self, request_id: str) -> List[Any]:
        """Pop all pending events for one request (non-blocking)."""
        with self._sink._lock:
            queue = self._sink._queues.get(request_id)
            if queue is None:
                return []
            out = list(queue)
            queue.clear()
        return out

    def wait(self, request_id: str, timeout: Optional[float] = None) -> bool:
        with self._lock:
            done = self._finished.get(request_id)
        if done is None:
            return True
        return done.wait(timeout=timeout)


class StreamChunk:
    """One structured output chunk (text + token ids + terminal facts)."""

    __slots__ = (
        "text",
        "delta_token_ids",
        "current_token_ids",
        "stopped",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "stop_sequence",
    )

    def __init__(
        self,
        text: str,
        delta_token_ids: List[int],
        current_token_ids: List[int],
        stopped: bool,
    ):
        self.text = text
        self.delta_token_ids = delta_token_ids
        self.current_token_ids = current_token_ids
        self.stopped = stopped
        self.finish_reason: Optional[str] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.stop_sequence: Optional[str] = None

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None


def map_finish_reason(reason: Optional[str]) -> str:
    """Internal event reasons → protocol-neutral finish vocabulary."""
    from astrai.inference.core import events as _events

    mapping = {
        _events.FINISH_STOP_TOKEN: "stop",
        _events.FINISH_LENGTH: "length",
        _events.FINISH_CANCELLED: "cancelled",
        _events.FINISH_ABORTED: "aborted",
        _events.FINISH_REJECTED: "rejected",
    }
    return mapping.get(reason or "", "stop")
