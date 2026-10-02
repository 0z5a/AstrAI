"""One-token advancement primitive shared by every scheduling mode."""

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

from astrai.extension import AttentionBackend, attn_backend
from astrai.inference.core.cache.pool import BlockPool, KVCacheManager
from astrai.inference.core.metrics import MetricsCollector
from astrai.inference.core.request import Request, RequestStatus
from astrai.inference.worker.model_runner import GPUModelRunner
from astrai.inference.worker.pending import PendingExecution


class SchedulerStep:
    """Advance every active request by one token (prefill + decode).

    Single shared primitive for both the continuous-batching loop and the
    synchronous ``run_batch`` path, so the two cannot drift.

    Tasks must already be allocated in the KV cache. Tasks without output
    are prefilled first and sample their first token from the final prompt
    position. Tasks with output extend the cache by one position and decode
    from their latest generated token.

    Decode runs as submit + commit: the executor launches the forward and
    sampling without resolving values on host, and the stepper's commit
    phase is the single place that appends tokens / logprobs to requests and
    advances their KV positions. ``step`` performs both back-to-back
    (synchronous semantics); ``step_submit`` / ``step_commit`` expose the
    halves for overlap scheduling.
    """

    def __init__(
        self,
        pool: BlockPool,
        cache_mgr: KVCacheManager,
        executor: GPUModelRunner,
        metrics: MetricsCollector,
        token_budget: Optional[int] = None,
    ):
        self._pool = pool
        self._kv_manager = cache_mgr
        self._executor = executor
        self._metrics = metrics
        # Max tokens per forward (prefill windows + decode steps).  None
        # disables chunking: every prefill runs its whole remaining prompt
        # in one forward (the historical behavior).
        self._token_budget = token_budget

    @staticmethod
    def _task_backend_groups(requests: List[Request]):
        groups = {}
        for request in requests:
            groups.setdefault(request.backend, (request.backend, []))[1].append(request)
        return groups.values()

    def step(
        self, requests: List[Request], return_logprobs: bool = False
    ) -> Tuple[List[Request], List[Request]]:
        """Advance ``requests`` by one token.

        Args:
            requests: Active requests to advance.
            return_logprobs: Forwarded to the executor; per-token logprobs
                are recorded on each request's ``output_logprobs``.

        Returns:
            ``(decoded, aborted)``: requests that produced a new token (its ID
            already appended to ``output_ids``) and requests that hit the
            sequence cap and were marked ``ABORTED``.
        """
        to_prefill = [t for t in requests if not t.prefill_complete and t.prompt_ids]
        prefilled_ids = set()
        # ``produced`` seeds with every LIVE request: callers drive their
        # live sets from it, and a request that only ran a continuation
        # chunk (or waited out the token budget) produced no token this
        # step but must stay live.  Aborted requests are filtered at the
        # return; finished ones are the caller's ``is_finished`` call.
        produced: List[Request] = list(requests)
        if to_prefill:
            for t in to_prefill:
                t.input_tokens = len(t.prompt_ids)

            groups: Dict[Tuple[int, Optional[AttentionBackend]], List[Request]] = {}
            for t in to_prefill:
                # Resume point = already-computed tokens (prefix hits pull
                # this forward on the first chunk; continuation chunks keep
                # advancing it via advance_kv).  Clamped to len-1 so the
                # final position always stays for the sampling chunk.
                start_pos = min(
                    max(
                        t.num_computed_tokens,
                        self._kv_manager.cached_tokens(t.request_id),
                    ),
                    len(t.prompt_ids) - 1,
                )
                groups.setdefault((start_pos, t.backend), []).append(t)

            budget = self._token_budget
            for (start_pos, _), group in groups.items():
                backend = group[0].backend
                backend_context = (
                    attn_backend(backend) if backend is not None else nullcontext()
                )
                # Chunked prefill: cap each forward at the token budget.
                # A group whose whole remaining prompts fit keeps the
                # historical single batched call; otherwise each request
                # contributes at most ONE window per step (continuation
                # chunks resume on the scheduler loop's next iteration),
                # so every call keeps a single shared start_pos — the
                # bind() contract.  The last window of a request samples
                # its token; earlier windows are KV-materialization-only
                # and advance the request's cursor without flipping
                # prefill_complete.
                windows: List[Tuple[List[Request], int, List[int]]] = []
                total = sum(len(t.prompt_ids) - start_pos for t in group)
                if budget is None or total <= budget:
                    windows.append(
                        (
                            group,
                            start_pos,
                            [len(t.prompt_ids) - start_pos for t in group],
                        )
                    )
                else:
                    used = 0
                    for t in group:
                        need = len(t.prompt_ids) - start_pos
                        take = min(need, budget - used)
                        if take <= 0:
                            break
                        windows.append(([t], start_pos, [take]))
                        used += take
                        if used >= budget:
                            break

                for chunk_reqs, w_start, chunk_lens in windows:
                    with (
                        backend_context,
                        self._metrics.record(
                            [t.request_id for t in chunk_reqs], "prefill"
                        ),
                    ):
                        prefilled, pending = self._executor.execute_prefill(
                            chunk_reqs,
                            start_pos=w_start,
                            return_logprobs=return_logprobs,
                            num_tokens=chunk_lens,
                        )

                    # Continuation chunks advance the KV cursor without
                    # completing prefill; final chunks flip the flag (the
                    # sampled token is not yet in KV — mark before commit
                    # so commit appends into consistent state).
                    for t, n in zip(chunk_reqs, chunk_lens):
                        if w_start + n >= len(t.prompt_ids):
                            t.mark_prefill_complete()
                        else:
                            t.advance_kv(w_start + n - t.num_computed_tokens)
                    if pending is not None:
                        self.step_commit(pending)
                    prefilled_ids.update(t.request_id for t in prefilled)

                    start_logical_page = w_start // self._pool.page_size
                    for t in chunk_reqs:
                        self._kv_manager.record_block_hashes(
                            t.request_id, t.prompt_ids, start_logical_page
                        )

        decoded: List[Request] = []
        aborted: List[Request] = []
        if prefilled_ids:
            # Mixed prefill/decode step: only requests whose prefill
            # COMPLETED (their first token is sampled and sitting in
            # output_ids) may decode.  Requests still awaiting their next
            # chunk (budget exhausted this step) are simply skipped — the
            # scheduler loop's next iteration resumes their windows.
            for t in requests:
                if t.request_id in prefilled_ids or not t.prefill_complete:
                    continue
                if self._kv_manager.extend_slots(t.request_id, t.next_pos):
                    decoded.append(t)
                else:
                    t.status = RequestStatus.ABORTED
                    aborted.append(t)
        else:
            decoded, aborted = self._extend_and_partition(requests)

        pending, produced_decoded, aborted_decoded = self._submit_decoded(
            decoded, return_logprobs, abort_on_refusal=True
        )
        aborted.extend(aborted_decoded)
        if pending is not None:
            self.step_commit(pending)

        # Every caller drives its live set from ``produced``; requests
        # whose window waited out this step's budget (or that only ran a
        # KV-materialization chunk) produced no token YET but must stay
        # live — the next step resumes them.  Terminal ones already sit in
        # ``aborted``.
        if aborted:
            aborted_ids = {t.request_id for t in aborted}
            produced = [t for t in produced if t.request_id not in aborted_ids]
        return produced, aborted

    def _extend_and_partition(
        self, requests: List[Request]
    ) -> Tuple[List[Request], List[Request]]:
        """Extend a step's requests with ONE batched cache call.

        The steady decode step (no prefills in the batch) is the hot
        path: per-request ``extend_slots`` costs one allocator round-trip
        each (~13us per request on serving-scale pools), the batched call
        harvests all new pages in one word-indexed pass.  Failure marks
        the request ABORTED exactly like the per-request loop.
        """
        ok = self._kv_manager.extend_slots_batch(
            [t.request_id for t in requests], [t.next_pos for t in requests]
        )
        decoded: List[Request] = []
        aborted: List[Request] = []
        for t, extended in zip(requests, ok):
            if extended:
                decoded.append(t)
            else:
                t.status = RequestStatus.ABORTED
                aborted.append(t)
        return decoded, aborted

    def _submit_decoded(
        self, decoded: List[Request], return_logprobs: bool, abort_on_refusal: bool
    ) -> Tuple[Optional[PendingExecution], List[Request], List[Request]]:
        """Submit every decode group; commit nothing.

        Shared by the synchronous ``step`` (which commits the returned
        pending immediately) and the overlap loop (which holds it for one
        scheduler iteration).  Refusal (frequency penalty with an
        uncommitted prior step) is retried after a drain; a second refusal
        aborts the group only in synchronous mode — the overlap caller
        drains before submitting, so refusal cannot reach it.

        KV positions advance HERE, at submit time: the write slot a step
        uses is a property of the launched work, not of its results, so
        the next submit can extend past an uncommitted step.  Commit only
        appends tokens/logprobs.
        """
        produced: List[Request] = []
        aborted: List[Request] = []
        pending: Optional[PendingExecution] = None
        for backend, group in self._task_backend_groups(decoded):
            backend_context = (
                attn_backend(backend) if backend is not None else nullcontext()
            )
            with (
                backend_context,
                self._metrics.record([t.request_id for t in group], "decode"),
            ):
                pending = self._executor.submit_decode(group, return_logprobs)
            if pending is None:
                # Refused combination (frequency penalty with an uncommitted
                # prior step): drain that step, then retry once.
                self._executor.flush_pending(self)
                with (
                    backend_context,
                    self._metrics.record([t.request_id for t in group], "decode"),
                ):
                    pending = self._executor.submit_decode(group, return_logprobs)
                if pending is None:
                    if abort_on_refusal:
                        for t in group:
                            t.status = RequestStatus.ABORTED
                            aborted.append(t)
                    continue
            for t in group:
                t.advance_kv()
            produced.extend(group)
        return pending, produced, aborted

    def step_submit(
        self, requests: List[Request], return_logprobs: bool = False
    ) -> Tuple[List[Request], Optional[PendingExecution]]:
        """Submit one step's work without committing results.

        The overlap scheduler's entry: prefill groups run through their
        usual submit path (their pending step is returned alongside the
        decode one — a batch mixing both yields at most one pending each
        per backend group; the last one wins and earlier ones must have
        been committed by ``flush_pending`` semantics).  Callers own the
        returned pending step and MUST commit it (or drain it via the
        executor) before touching the requests again.
        """
        to_prefill = [t for t in requests if not t.prefill_complete and t.prompt_ids]
        prefilled_ids = set()
        produced: List[Request] = []
        pendings: List[PendingExecution] = []
        if to_prefill:
            for t in to_prefill:
                t.input_tokens = len(t.prompt_ids)

            groups: Dict[Tuple[int, Optional[AttentionBackend]], List[Request]] = {}
            for t in to_prefill:
                start_pos = min(
                    self._kv_manager.cached_tokens(t.request_id), len(t.prompt_ids) - 1
                )
                groups.setdefault((start_pos, t.backend), []).append(t)

            for (start_pos, _), group in groups.items():
                backend = group[0].backend
                backend_context = (
                    attn_backend(backend) if backend is not None else nullcontext()
                )
                with (
                    backend_context,
                    self._metrics.record([t.request_id for t in group], "prefill"),
                ):
                    prefilled, pending = self._executor.execute_prefill(
                        group, start_pos=start_pos, return_logprobs=return_logprobs
                    )
                if pending is not None:
                    pending.prefill_request_ids = tuple(t.request_id for t in prefilled)
                    pendings.append(pending)
                prefilled_ids.update(t.request_id for t in prefilled)
                produced.extend(prefilled)

                start_logical_page = start_pos // self._pool.page_size
                for t in group:
                    self._kv_manager.record_block_hashes(
                        t.request_id, t.prompt_ids, start_logical_page
                    )

        decoded: List[Request] = []
        if prefilled_ids:
            for t in requests:
                if t.request_id in prefilled_ids:
                    continue
                if self._kv_manager.extend_slots(t.request_id, t.next_pos):
                    decoded.append(t)
        else:
            decoded, _aborted = self._extend_and_partition(requests)

        if decoded:
            pending, produced_decoded, _ = self._submit_decoded(
                decoded, return_logprobs, abort_on_refusal=False
            )
            if pending is not None:
                pendings.append(pending)
            produced.extend(produced_decoded)

        # A step may hold at most one uncommitted pending per backend
        # group; in the steady decode loop it is exactly one.  Mixed
        # prefill+decode steps (batch change) commit inline instead — the
        # overlap loop drains before re-batching anyway.
        if len(pendings) == 1:
            return produced, pendings[0]
        for extra in pendings:
            self.step_commit(extra)
        return produced, None

    def step_commit(self, pending: PendingExecution) -> List[Request]:
        """Materialise a submitted step onto its requests.

        The single commit entry point: appends each sampled token (and its
        logprob, when recorded) to the request's output state and advances the
        KV write position.  Idempotent — a step already committed is a
        no-op — so abort paths can call it defensively.  Prefill callers
        mark ``prefill_complete`` before this (the first token is already in
        the request's history at that point).

        A request that already reached its ``max_tokens`` when this step was
        in flight drops the extra token: the overlap pipeline may launch
        one step past the terminal one, and user-visible output must never
        run past the termination point.
        """
        if pending.committed:
            return []
        payload = pending.commit()
        produced: List[Request] = []
        preflip = pending.prefill_request_ids
        for request, (token_id, logprob) in zip(pending.requests, payload):
            if request.request_id in preflip:
                request.mark_prefill_complete()
            if (
                request.max_tokens is not None
                and request.output_tokens >= request.max_tokens
            ):
                continue
            request.output_ids.append(token_id)
            request.output_tokens += 1
            if logprob is not None:
                request.output_logprobs.append(logprob)
            produced.append(request)
        return produced
