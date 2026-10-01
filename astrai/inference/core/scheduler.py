import logging
import threading
import uuid
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch

from astrai.config.inference_config import InferenceConfig
from astrai.extension import (
    ATTN_BACKEND,
    AttentionBackend,
    attn_backend,
    get_backend,
)
from astrai.inference.core import request as request_module
from astrai.inference.core.cache.pool import BlockPool, KVCacheManager
from astrai.inference.core.events import (  # noqa: I001 module path, not the frontend package
    FINISH_ABORTED,
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_REJECTED,
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.core.metrics import MetricsCollector
from astrai.inference.core.request import (
    STOP,
    GenerationResult,
    Request,
    RequestManager,
    RequestStatus,
    StreamDecoder,
)
from astrai.inference.core.stepper import SchedulerStep
from astrai.inference.core.versioning import PolicyVersionGuard
from astrai.inference.worker.model_runner import GPUModelRunner
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)
T = TypeVar("T")
_config = InferenceConfig()


def _with_weight_lock(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._weight_lock:
            return method(self, *args, **kwargs)

    return synchronized


class OutputEventSink:
    """Callable receiving one step's worth of output events.

    Contract: invoked on the scheduler loop thread; implementations must
    be fast, non-blocking and exception-safe (failures are logged and the
    batch dropped — the loop keeps running).
    """

    def __call__(self, events: List[Any]) -> None:
        raise NotImplementedError


class _CallbackBridge(OutputEventSink):
    """Default sink: maps output events onto registered stream callbacks.

    Keeps the pre-event behavior for consumers that registered plain or
    batched callbacks on the RequestManager: ``TokenDelta`` is detokenized
    here (sink side, not the loop's emission code) and delivered as the
    incremental text, terminal events become the ``STOP`` sentinel.  The
    per-request decoder state lives on the bridge, mirroring where the
    request's ``StreamDecoder`` used to live.
    """

    def __init__(self, requests: RequestManager):
        self._requests = requests
        self._tokenizer = requests.tokenizer
        self._decoders: Dict[str, StreamDecoder] = {}

    def __call__(self, events: List[Any]) -> None:
        payload: List[Tuple[str, Any]] = []
        for event in events:
            if isinstance(event, TokenDelta):
                decoder = self._decoders.get(event.request_id)
                if decoder is None:
                    # Module-attribute lookup so tests (and future
                    # overrides) that patch ``request.StreamDecoder``
                    # remain effective here.
                    decoder = self._decoders[event.request_id] = (
                        request_module.StreamDecoder(self._tokenizer)
                    )
                text = decoder.push(event.token_id)
                if text:
                    payload.append((event.request_id, text))
            elif isinstance(event, (RequestFinished, RequestError)):
                payload.append((event.request_id, STOP))
                self._decoders.pop(event.request_id, None)
        if payload:
            self._requests.invoke_callbacks(payload)


class Scheduler:
    """Continuous batching loop: cleanup -> refill -> prefill -> decode (all groups)."""

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache: Optional[BlockPool] = None,
        enable_cuda_graph: bool = True,
        backend: Optional[Union[str, ATTN_BACKEND, AttentionBackend, type]] = None,
        policy_version: int = 0,
        enable_overlap: bool = False,
        page_size: Optional[int] = None,
        kv_tokens: Optional[int] = None,
        token_budget: Optional[int] = None,
    ):
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise ValueError("policy_version must be a non-negative integer")
        # Depth-2 submit/commit overlap for steady online decode batches.
        # Keep opt-in because its benefit depends on serving workload; the
        # contract is exercised regardless, while run_batch stays synchronous.
        self._enable_overlap = enable_overlap
        config = model.config

        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        elif config.max_position_embeddings is not None:
            self.max_seq_len = config.max_position_embeddings
        else:
            raise ValueError(
                "max_seq_len must be provided either as argument "
                "or in model config (config.max_position_embeddings)"
            )
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        head_dim = config.hidden_size // config.num_attention_heads

        if cache is not None:
            self._cache = cache
        else:
            # page_size/kv_tokens select the paged strategy with prefix
            # caching; the default stays contiguous (static partitions),
            # which rollout-sized serving keeps as the zero-config path.
            pool_kwargs: Dict[str, Any] = {}
            if page_size is not None:
                pool_kwargs["page_size"] = page_size
            if kv_tokens is not None:
                pool_kwargs["n_tokens"] = kv_tokens
            self._cache = BlockPool(
                n_layers=config.num_hidden_layers,
                n_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=self.max_seq_len,
                device=self.device,
                dtype=self.dtype,
                **pool_kwargs,
            )

        self._metrics = MetricsCollector()

        self._kv_manager = KVCacheManager(self._cache)

        self._requests = RequestManager(
            tokenizer=tokenizer,
            max_batch_size=max_batch_size,
            max_seq_len=self.max_seq_len,
            metrics=self._metrics,
        )

        if backend is None:
            self._backend = None
            active_backend = get_backend()
        else:
            active_backend = backend
        with attn_backend(active_backend):
            if backend is not None:
                self._backend = get_backend()
            self._backend_name = type(get_backend()).__name__
            self._executor = GPUModelRunner(
                model=model,
                kv_cache=self._cache,
                cache_mgr=self._kv_manager,
                device=self.device,
                dtype=self.dtype,
                enable_cuda_graph=enable_cuda_graph,
            )

        # 0/None both mean "no chunking": whole-remaining-prompt prefills.
        effective_budget = token_budget or _config.max_num_batched_tokens or None
        self._stepper = SchedulerStep(
            self._cache,
            self._kv_manager,
            self._executor,
            self._metrics,
            token_budget=effective_budget,
        )

        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None
        # Finished requests whose KV slots are still written by the in-flight
        # step; freed once that step is committed (see run_busy_loop).
        self._retired: List[Request] = []
        # Output event sink.  The default bridge maps events onto the
        # RequestManager's registered callbacks (legacy behavior); the
        # engine installs a bounded-queue sink so the loop never runs
        # consumer code.  Swappable per deployment (T1: cross-process).
        self._event_sink = _CallbackBridge(self._requests)
        self._policy_guard = PolicyVersionGuard(
            policy_version,
            ensure_ready=self._ensure_weight_update_ready,
            on_commit=self._kv_manager.invalidate_cache,
        )
        # Synchronous generation shares the guard's generation/weight mutex.
        self._weight_lock = self._policy_guard.lock

    def set_event_sink(self, sink: "OutputEventSink") -> None:
        """Route scheduler output events to ``sink`` (idempotent swap)."""
        self._event_sink = sink

    def _emit_events(self, events: List[Any]) -> None:
        try:
            self._event_sink(events)
        except Exception:
            # A consumer failure must never take down the engine loop;
            # the offending request's terminal event still reaches its
            # queue via the sink's own error handling.
            logger.exception("output event sink failed; dropping batch")

    @property
    def policy_version(self) -> int:
        """Version of the model weights used for subsequent generations."""
        return self._policy_guard.policy_version

    def _ensure_weight_update_ready(self) -> None:
        """Check weight update preconditions. Must be called under the lock."""
        if self._loop_thread is not None and self._loop_thread.is_alive():
            raise RuntimeError("Stop the scheduler before updating model weights")
        if (
            self._requests.get_running_requests()
            or self._requests.get_waiting_requests()
        ):
            raise RuntimeError("Cannot update model weights while requests are queued")
        # Drain any submitted-but-uncommitted step: its KV writes and
        # device references must land before the world changes underneath.
        self._executor.flush_pending(self._stepper)

    def update_weights(self, policy_version: int) -> int:
        """Acknowledge an in-place weight update and invalidate stale KV state.

        The scheduler owns the same model object as the in-process trainer, so
        weights have already changed when this method is called. The explicit
        version update makes that lifecycle visible and prevents prefix KV
        entries produced by older weights from being reused.
        """
        return self._policy_guard.update_weights(policy_version)

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        """Mutate shared weights and publish their version without generation.

        ``policy_version=None`` derives ``live + 1`` under the same lock, for
        callers that only need "advance by one" (e.g. ``optimizer.step()``)
        without a read-compute-write race on the current version.  The
        derived target version is handed to ``update``.
        """
        return self._policy_guard.apply_weight_update(policy_version, update)

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        """Inspect state while the scheduler's policy version remains stable."""
        return self._policy_guard.with_policy_snapshot(inspect)

    def add_request(self, prompt: str, **kwargs) -> str:
        return self._requests.add_request(prompt, **kwargs)

    def add_requests(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch add; see RequestManager (ids/pre-tokenized passthrough)."""
        return self._requests.add_requests(prompts, **kwargs)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a waiting or active request without freeing in-use KV state."""
        immediate, cancelled = self._requests.cancel_request(request_id)
        for request in immediate:
            self._metrics.mark_finished(
                request.request_id, request.input_tokens, request.output_tokens
            )
        if cancelled:
            self._requests.wake()
        return cancelled

    def remove_request(self, request_id: str) -> bool:
        """Backward-compatible alias for cancellation."""
        return self.cancel_request(request_id)

    def get_stats(self) -> Dict[str, Any]:
        stats = self._requests.get_stats()
        stats["kv_cache_tasks"] = self._kv_manager.request_count
        stats["policy_version"] = self._policy_guard.policy_version
        return stats

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._executor.cuda_graph_enabled

    def _backend_context(self):
        if self._backend is None:
            return nullcontext()
        return attn_backend(self._backend)

    def _step(
        self, requests: List[Request], return_logprobs: bool = False
    ) -> Tuple[List[Request], List[Request]]:
        """Advance every active request by one token; see :class:`SchedulerStep`."""
        return self._stepper.step(requests, return_logprobs=return_logprobs)

    def run_busy_loop(self):
        # Set membership is O(1); the tokenizer rebuilds the list on every
        # attribute access, and both the finished-request scan and the
        # per-step terminal check probe it (×2 per request per step before).
        stop_ids = frozenset(self._requests.tokenizer.stop_ids)
        try:
            with self._backend_context():
                while not self._stop_event.is_set():
                    finished = self._requests.remove_finished_requests(stop_ids)
                    retired = getattr(self, "_retired", None)
                    if finished:
                        # A finished request whose KV is still written by the
                        # in-flight (uncommitted) step must not free its
                        # slots yet — the slots could be reallocated and
                        # corrupted mid-flight.  Such requests are deferred to
                        # the next iteration's drain (the pending step is
                        # always committed before a batch change there).
                        pending = self._executor.peek_pending()
                        inflight_ids = (
                            set(pending.snapshot.request_ids)
                            if pending is not None and not pending.committed
                            else frozenset()
                        )
                        for request in finished:
                            if request.status == RequestStatus.FINISHED:
                                self._kv_manager.record_block_hashes(
                                    request.request_id,
                                    self._kv_manager.request_cacheable_ids(
                                        request.request_id,
                                        request.prompt_ids,
                                        request.output_ids,
                                    )
                                    if self._cache.page_size > 1
                                    else request.prompt_ids,
                                )
                            if (
                                retired is not None
                                and request.request_id in inflight_ids
                            ):
                                retired.append(request)
                            else:
                                self._kv_manager.free_slots(request.request_id)

                    if retired:
                        pending = self._executor.peek_pending()
                        if pending is None or pending.committed:
                            for request in retired:
                                self._kv_manager.free_slots(request.request_id)
                            retired.clear()

                    active = self._requests.get_running_requests()
                    available = self._requests.max_batch_size - len(active)
                    if available > 0:
                        candidates = self._requests.pull_waiting(available)
                        failed = []
                        hopeless = []
                        for request in candidates:
                            if not self._kv_manager.can_ever_fit(
                                len(request.prompt_ids)
                            ):
                                # Larger than the whole pool even with every
                                # cached page evicted: retrying would spin
                                # the allocator forever (the livelock), so
                                # terminate the request instead.
                                hopeless.append(request)
                                continue
                            if self._kv_manager.alloc_slots(
                                request.request_id, request.prompt_ids
                            ):
                                if not self._requests.activate(request):
                                    self._kv_manager.free_slots(request.request_id)
                                    self._metrics.mark_finished(
                                        request.request_id,
                                        request.input_tokens,
                                        request.output_tokens,
                                    )
                                else:
                                    # Just activated: extend the snapshot
                                    # taken above instead of re-listing the
                                    # whole active set a second time.
                                    active.append(request)
                            else:
                                failed.append(request)
                        if hopeless:
                            # Terminal events first (consumers unblock),
                            # then release their queue records.
                            self._emit_events(
                                [
                                    RequestFinished(
                                        request_id=request.request_id,
                                        finish_reason=FINISH_REJECTED,
                                        prompt_tokens=len(request.prompt_ids),
                                    )
                                    for request in hopeless
                                ]
                            )
                            for request in hopeless:
                                self._metrics.mark_finished(
                                    request.request_id,
                                    len(request.prompt_ids),
                                    0,
                                )
                            self._requests.discard_waiting(hopeless)
                        if failed:
                            self._requests.return_to_waiting(failed)

                    if not active:
                        # Idle path: drain any residual step and release
                        # deferred retirements BEFORE parking, so the loop
                        # never waits with an uncommitted step in flight.
                        self._executor.flush_pending(self._stepper)
                        if retired:
                            for request in retired:
                                self._kv_manager.free_slots(request.request_id)
                            retired.clear()
                        if not self._requests.has_requests():
                            self._requests.wait_for_requests(timeout=1.0)
                            continue
                        # Refill was rejected (KV pressure): re-check after
                        # waiting so a slot freed elsewhere is picked up.
                        active = [
                            request
                            for request in self._requests.get_running_requests()
                            if request.status != RequestStatus.ABORTED
                        ]

                    # Drop any ABORTED members (status can flip during a
                    # step) before stepping.
                    active = [
                        request
                        for request in active
                        if request.status != RequestStatus.ABORTED
                    ]

                    # ---- overlap pipeline (depth 2) ----
                    # Submit the current step first, then commit the step
                    # submitted one iteration ago: the commit's host work
                    # (tolist wait, decode, callbacks) then overlaps the
                    # GPU computing the step just launched.  Only steady
                    # decode batches ride the pipeline — a changed batch
                    # (finish, join, prefill mix) drains first and falls
                    # back to the synchronous step, because the next
                    # submit's inputs depend on committed request state.
                    pending = self._executor.peek_pending()
                    aborted: List[Request] = []
                    overlap = self._enable_overlap
                    steady = (
                        overlap
                        and active
                        and pending is not None
                        and pending.snapshot.request_ids
                        == tuple(t.request_id for t in active)
                        and all(t.prefill_complete for t in active)
                        and self._executor.can_overlap_submit()
                    )
                    if steady:
                        # step_submit owns the executor's pending slot: the
                        # step it launches replaces the in-flight one, so the
                        # slot must NOT be cleared again here — clearing it
                        # would drop the just-submitted step (its tokens then
                        # never commit; every other steady iteration lost a
                        # token and requests aborted at the KV cap instead of
                        # max_tokens).
                        produced, new_pending = self._stepper.step_submit(active)
                        committed = self._stepper.step_commit(pending)
                    else:
                        self._executor.flush_pending(self._stepper)
                        produced, aborted = self._stepper.step(active)
                        new_pending = None
                        committed = produced

                    decoded = [
                        t for t in committed if t.status != RequestStatus.ABORTED
                    ]

                    # Event emission: token ids + terminal facts only.  The
                    # loop thread never detokenizes and never runs user
                    # callbacks — the sink decides where the events land
                    # (bounded queue for the engine, direct callbacks for
                    # legacy consumers, no-op when nobody listens).
                    events: List[Any] = [
                        RequestFinished(
                            request_id=t.request_id,
                            finish_reason=FINISH_ABORTED,
                        )
                        for t in aborted
                    ]
                    for t in decoded:
                        if t.status == RequestStatus.ABORTED:
                            continue
                        if not t.output_ids:
                            # Defensive: a decoded request with no committed
                            # token yet (cannot happen on the contract's
                            # happy path) must not crash the loop.
                            continue
                        events.append(
                            TokenDelta(
                                request_id=t.request_id,
                                token_id=t.output_ids[-1],
                                sequence_no=t.output_tokens,
                            )
                        )
                        if t.is_finished(stop_ids):
                            reason = (
                                FINISH_STOP_TOKEN
                                if t.output_ids[-1] in stop_ids
                                else FINISH_LENGTH
                            )
                            events.append(
                                RequestFinished(
                                    request_id=t.request_id,
                                    finish_reason=reason,
                                    prompt_tokens=t.input_tokens,
                                    completion_tokens=t.output_tokens,
                                )
                            )
                    if events:
                        self._emit_events(events)

        except Exception as e:
            self._stop_event.set()
            logger.error(f"Scheduler loop crashed: {e}", exc_info=True)
            self._executor.flush_pending(self._stepper)
            self._abort_and_clear(free_waiting=False)

    def start(self):
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._stop_event.clear()
        t = threading.Thread(target=self.run_busy_loop, daemon=True)
        t.start()
        self._loop_thread = t

    def stop(self):
        self._stop_event.set()
        self._requests.wake()
        thread = self._loop_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            # The loop did not drain in time (a stuck forward, a slow
            # callback).  Clearing queues underneath a live loop would
            # double-free KV slots and let a second start() race it, so
            # leave the thread handle in place: callers can retry stop(),
            # and start() refuses to launch a second loop while it lives.
            logger.warning(
                "scheduler loop did not stop within 2s; keeping thread "
                "handle and request state — call stop() again once the "
                "blocking work drains"
            )
            return
        self._loop_thread = None
        self._abort_and_clear(free_waiting=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _abort_and_clear(self, free_waiting: bool):
        """Emit terminal events, release cache slots, and clear request queues."""
        active = self._requests.get_running_requests()
        waiting = self._requests.get_waiting_requests()
        terminal = [
            RequestFinished(
                request_id=request.request_id,
                finish_reason=FINISH_CANCELLED,
                prompt_tokens=request.input_tokens,
                completion_tokens=request.output_tokens,
            )
            for request in (*active, *waiting)
        ]
        if terminal:
            self._emit_events(terminal)
        for request in active:
            self._kv_manager.free_slots(request.request_id)
            self._metrics.mark_finished(
                request.request_id, request.input_tokens, request.output_tokens
            )
        for request in waiting:
            if free_waiting:
                self._kv_manager.free_slots(request.request_id)
            self._metrics.mark_finished(
                request.request_id, request.input_tokens, request.output_tokens
            )
        self._requests.clear_queues()

    @_with_weight_lock
    def run_batch(
        self,
        prompt_ids_list: List[List[int]],
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
        return_logprobs: bool = False,
        return_details: bool = False,
    ) -> List[Any]:
        """Synchronous batch generation without the scheduler thread.

        Accepts already-tokenized prompts (no string round-trip) and runs
        prefill + decode to completion on the calling thread.  Designed for
        RL rollout, where logprobs of the behaviour policy must be collected
        alongside generated tokens.

        Args:
            prompt_ids_list: ``B`` prompts, each a list of token IDs.
            max_tokens: Maximum tokens to generate per prompt.  ``None``
                uses ``self.max_seq_len - len(prompt_ids)``.
            temperature/top_p/top_k/frequency_penalty/rep_window: Sampling
                parameters (uniform across the batch).
            return_logprobs: If ``True``, return ``(token_ids, logprobs)``
                tuples per prompt (logprobs aligned 1-to-1 with token_ids).
            return_details: If ``True``, return a structured result per prompt
                with terminal and error reasons. Logprobs are populated when
                ``return_logprobs`` is also ``True``.

        Returns:
            Structured results when ``return_details`` is ``True``;
            otherwise generated token IDs per prompt, or token/logprob tuples
            when ``return_logprobs`` is ``True``.
        """
        stop_ids = self._requests.tokenizer.stop_ids
        seq_cap = self.max_seq_len
        request_backend = get_backend(use_default=False)

        requests: List[Optional[Request]] = []
        error_reasons: List[Optional[str]] = []
        for ids in prompt_ids_list:
            if not ids:
                requests.append(None)
                error_reasons.append("prompt_empty")
                continue
            if len(ids) >= seq_cap:
                requests.append(None)
                error_reasons.append("prompt_too_long")
                continue
            t_max = max_tokens
            if t_max is None:
                t_max = seq_cap - len(ids)
            else:
                t_max = min(t_max, seq_cap - len(ids))
            if t_max <= 0:
                requests.append(None)
                error_reasons.append("max_tokens_non_positive")
                continue
            request = Request(
                request_id=f"batch_{uuid.uuid4().hex[:8]}",
                prompt_ids=list(ids),
                max_tokens=t_max,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                rep_window=rep_window,
                backend=request_backend,
            )
            if not self._kv_manager.alloc_slots(request.request_id, request.prompt_ids):
                requests.append(None)
                error_reasons.append("kv_cache_allocation_failed")
                continue
            request.input_tokens = len(request.prompt_ids)
            self._metrics.register(request.request_id)
            requests.append(request)
            error_reasons.append(None)

        runtime_errors: Dict[str, str] = {}
        try:
            live = [t for t in requests if t is not None]

            with self._backend_context():
                while live:
                    decoded, aborted = self._stepper.step(
                        live, return_logprobs=return_logprobs
                    )
                    for request in aborted:
                        runtime_errors[request.request_id] = "kv_cache_extension_failed"
                    live = [t for t in decoded if not t.is_finished(stop_ids)]
        finally:
            for t in requests:
                if t is not None:
                    self._metrics.mark_finished(
                        t.request_id, t.input_tokens, t.output_tokens
                    )
                    self._kv_manager.free_slots(t.request_id)

        details: List[GenerationResult] = []
        for t, setup_error in zip(requests, error_reasons):
            if t is None:
                details.append(
                    GenerationResult(
                        token_ids=[],
                        logprobs=[],
                        finish_reason="rejected",
                        error_reason=setup_error,
                    )
                )
            else:
                runtime_error = runtime_errors.get(t.request_id)
                stopped = bool(t.output_ids and t.output_ids[-1] in stop_ids)
                if runtime_error:
                    finish_reason = "rejected"
                elif stopped:
                    finish_reason = "stop"
                else:
                    finish_reason = "length"
                details.append(
                    GenerationResult(
                        token_ids=list(t.output_ids),
                        logprobs=list(t.output_logprobs),
                        finish_reason=finish_reason,
                        error_reason=runtime_error,
                    )
                )

        if return_details:
            return details
        if return_logprobs:
            return [(result.token_ids, result.logprobs) for result in details]
        return [result.token_ids for result in details]

    def score_ids(
        self,
        prompt_ids_list: List[List[int]],
        continuation_ids_list: List[List[int]],
        per_token: bool = False,
    ) -> List[Any]:
        """Teacher-forced log-probabilities, one entry per request.

        Unlike :meth:`generate` nothing is sampled: each request is fed
        ``prompt + continuation`` and only the continuation's own tokens are
        scored, so the caller gets P(continuation | prompt) under the model.
        This is the single entry point for log-likelihood metrics (MMLU,
        HellaSwag, perplexity, IFD), which used to each drive the model with
        their own attention mask.

        Args:
            prompt_ids_list: ``B`` contexts.
            continuation_ids_list: ``B`` continuations, each non-empty and
                shorter than the concatenated sequence.
            per_token: return per-token log-probabilities instead of the sum.

        Returns:
            ``List[float]`` of summed log-probabilities, or ``List[List[float]]``
            when ``per_token`` is ``True``.  A request that cannot be scored
            (empty, or the whole sequence at the sequence cap) yields ``None``.
        """
        if len(prompt_ids_list) != len(continuation_ids_list):
            raise ValueError("prompt and continuation lists must have equal length")

        request_backend = get_backend(use_default=False)
        seq_cap = self.max_seq_len
        requests: List[Optional[Request]] = []
        for prompt_ids, cont_ids in zip(prompt_ids_list, continuation_ids_list):
            if not prompt_ids or not cont_ids:
                requests.append(None)
                continue
            scored_ids = list(prompt_ids) + list(cont_ids)
            if len(scored_ids) > seq_cap:
                requests.append(None)
                continue
            request = Request(
                request_id=f"score_{uuid.uuid4().hex[:8]}",
                prompt_ids=scored_ids,
                max_tokens=0,
                backend=request_backend,
            )
            request.cont_len = len(cont_ids)
            if not self._kv_manager.alloc_slots(request.request_id, request.prompt_ids):
                requests.append(None)
                continue
            request.input_tokens = len(request.prompt_ids)
            requests.append(request)

        live = [t for t in requests if t is not None]
        results: Dict[str, Any] = {}
        try:
            if live:
                with self._backend_context():
                    scored = self._executor.execute_score(live, per_token=per_token)
                for request, value in zip(live, scored):
                    results[request.request_id] = value
        finally:
            for request in live:
                self._kv_manager.free_slots(request.request_id)

        return [
            results.get(request.request_id) if request is not None else None
            for request in requests
        ]
