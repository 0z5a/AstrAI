"""submit/commit contract for one decoded step.

The executor's ``submit`` path launches the model forward, sampling and
result relay without resolving a single device value on the host and
without mutating any :class:`~astrai.inference.task.Task`.  Its product is
a :class:`PendingStep` — a message-shaped handle that names the batch
(snapshot), holds the device-resident sampled tokens and the ordered task
references, and exposes the one sanctioned host materialisation point
(``commit``).

This is the process-wall seam: when the executor eventually lives behind
a transport (multi-GPU broadcast, out-of-process workers), the message
that crosses it is exactly this pending step, and ``commit`` remains the
single consumer of results on the scheduling side.

The result ring (``ResultRing``) is the async-D2H twin of the pending
step: submit posts the device tokens (and optional logprobs) into a
pinned host slot on a dedicated copy stream and records a completion
event; commit then waits on that event instead of a blocking
``tolist``.  Slot reuse is guarded by the previous occupant's event, so
an in-flight copy is never overwritten.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class BatchSnapshot:
    """Frozen identity of the batch a pending step belongs to.

    ``task_ids`` preserves submit-time order; ``kv_positions`` is the
    pre-forward next-write position per slot (host bookkeeping, valid
    regardless of when the commit runs).  ``policy_version`` records the
    weight version the forward ran under so a late commit can detect that
    the world moved (weights updated, cache invalidated) and refuse
    instead of corrupting state.
    """

    task_ids: Tuple[str, ...]
    kv_positions: Tuple[int, ...]
    policy_version: int


@dataclass
class PendingStep:
    """One submitted decode step awaiting its commit.

    Held fields:

    - ``tokens``: sampled token ids ``[B]`` on device.  Kept alive here and
      by the executor's steady-state relay (``last_tokens``); commit reads
      it exactly once — via the async copy event when the result ring
      posted one, else via ``tolist``.
    - ``logprobs``: optional ``[B]`` device tensor of chosen-token logprobs
      under the raw model distribution (``return_logprobs`` batches).
    - ``tasks``: the ordered task references.  Commit validates the world
      still matches the snapshot before touching them.
    """

    snapshot: BatchSnapshot
    tasks: List[object]
    tokens: Tensor
    logprobs: Optional[Tensor] = None
    # Tasks whose prompt KV this step materialised: their ``prefill_done``
    # flag flips only at commit time (the first token must be appended
    # first), so the flag never advertises a state the task has not
    # reached for consumers that read host output history.
    prefill_task_ids: Tuple[str, ...] = ()
    copy_event: Optional["torch.cuda.Event"] = None
    host_tokens: Optional[Tensor] = None
    host_logprobs: Optional[Tensor] = None
    _ring: Optional["ResultRing"] = field(default=None, repr=False)

    def commit(self) -> List[Tuple[int, Optional[float]]]:
        """Materialise results on host; the sanctioned D2H of this step.

        Waits on the posted copy event (async path) or falls back to
        ``tolist`` (no ring / CPU).  Idempotent: the resolved payload is
        cached and returned on repeat calls, so an abort path that already
        consumed the step cannot double-append tokens to tasks.
        """
        if self._payload is None:
            if self.copy_event is not None and self.host_tokens is not None:
                self.copy_event.synchronize()
                tokens_list = self.host_tokens.tolist()
                logprobs_list = (
                    self.host_logprobs.tolist()
                    if self.host_logprobs is not None
                    else None
                )
                if self._ring is not None:
                    self._ring.release(self)
            else:
                tokens_list = self.tokens.tolist()
                logprobs_list = (
                    self.logprobs.tolist() if self.logprobs is not None else None
                )
            if logprobs_list is not None:
                self._payload = list(zip(tokens_list, logprobs_list))
            else:
                self._payload = [(t, None) for t in tokens_list]
        return self._payload

    @property
    def committed(self) -> bool:
        return self._payload is not None

    _payload: Optional[List[Tuple[int, Optional[float]]]] = field(
        default=None, init=False, repr=False
    )


class ResultRing:
    """Depth-2 pinned host slots fed by a dedicated copy stream.

    ``post`` stages a pending step's device results into the free slot:
    the copy stream first waits on the producing work (current stream
    ordering is inherited — tokens were produced on the compute stream),
    copies ``[B]`` tokens (and optional ``[B]`` logprobs) into pinned
    host buffers and records a completion event onto the pending step.
    Depth 2 lets the scheduler submit step t+1's copy while step t's copy
    may still be in flight; a slot is only reused once its previous
    occupant's event has been waited on (the pending step itself holds
    the only reference that matters, and ``post`` synchronises the old
    event before overwriting).

    On CPU (or when CUDA is unavailable) the ring is inert and pending
    steps fall back to synchronous ``tolist`` commits.
    """

    def __init__(self, max_batch_size: int, device):
        self._enabled = (
            torch.cuda.is_available() and torch.device(device).type == "cuda"
        )
        self._slots = []
        if self._enabled:
            self._copy_stream = torch.cuda.Stream()
            for _ in range(2):
                self._slots.append(
                    {
                        "tokens": torch.empty(
                            max_batch_size, dtype=torch.long, pin_memory=True
                        ),
                        "logprobs": torch.empty(
                            max_batch_size, dtype=torch.float32, pin_memory=True
                        ),
                        "event": torch.cuda.Event(),
                        "in_use": False,
                    }
                )
        else:
            self._copy_stream = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def post(self, pending: PendingStep) -> bool:
        """Stage ``pending``'s results into a ring slot; False if refused.

        Refusal means the caller should keep the synchronous ``tolist``
        fallback (used for CPU runs and for oversized batches).
        """
        if not self._enabled:
            return False
        b = pending.tokens.shape[0]
        if b > self._slots[0]["tokens"].shape[0]:
            return False
        slot = None
        for candidate in self._slots:
            if not candidate["in_use"]:
                slot = candidate
                break
        if slot is None:
            # Both slots busy: wait for the oldest occupant's event before
            # overwriting — the copy is finished or the wait is the cost
            # of running more than two steps behind, never a corruption.
            slot = self._slots[0]
            slot["event"].synchronize()
        tokens_pin = slot["tokens"][:b]
        logprobs_pin = slot["logprobs"][:b] if pending.logprobs is not None else None
        # The copy stream must not race the producing work: tokens were
        # sampled on the compute stream (possibly still in flight). Record
        # the current stream position and make the copy stream wait on it
        # before the D2H, so the pinned slot never lands a stale read.
        produced = torch.cuda.Event()
        produced.record()
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(produced)
            tokens_pin.copy_(pending.tokens, non_blocking=True)
            if logprobs_pin is not None:
                logprobs_pin.copy_(pending.logprobs, non_blocking=True)
            slot["event"].record()
        slot["in_use"] = True
        pending.copy_event = slot["event"]
        pending.host_tokens = tokens_pin
        pending.host_logprobs = logprobs_pin
        pending._ring = self
        return True

    def release(self, pending: PendingStep) -> None:
        """Mark the slot a committed step occupied as reusable."""
        if not self._enabled:
            return
        for slot in self._slots:
            if slot["event"] is pending.copy_event:
                slot["in_use"] = False
                break
