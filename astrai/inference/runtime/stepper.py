"""One-token advancement primitive shared by every scheduling mode."""

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

from astrai.extension import AttentionBackend, attn_backend
from astrai.inference.cache import PagePool, TaskCacheManager
from astrai.inference.metrics import MetricsCollector
from astrai.inference.runtime.model_runner import GPUModelRunner
from astrai.inference.runtime.pending import PendingStep
from astrai.inference.task import Task, TaskStatus


class Stepper:
    """Advance every active task by one token (prefill + decode).

    Single shared primitive for both the continuous-batching loop and the
    synchronous ``run_batch`` path, so the two cannot drift.

    Tasks must already be allocated in the KV cache. Tasks without output
    are prefilled first and sample their first token from the final prompt
    position. Tasks with output extend the cache by one position and decode
    from their latest generated token.

    Decode runs as submit + commit: the executor launches the forward and
    sampling without resolving values on host, and the stepper's commit
    phase is the single place that appends tokens / logprobs to tasks and
    advances their KV positions. ``step`` performs both back-to-back
    (synchronous semantics); ``step_submit`` / ``step_commit`` expose the
    halves for overlap scheduling.
    """

    def __init__(
        self,
        pool: PagePool,
        task_cache: TaskCacheManager,
        executor: GPUModelRunner,
        metrics: MetricsCollector,
    ):
        self._pool = pool
        self._task_cache = task_cache
        self._executor = executor
        self._metrics = metrics

    @staticmethod
    def _task_backend_groups(tasks: List[Task]):
        groups = {}
        for task in tasks:
            groups.setdefault(task.backend, (task.backend, []))[1].append(task)
        return groups.values()

    def step(
        self, tasks: List[Task], return_logprobs: bool = False
    ) -> Tuple[List[Task], List[Task]]:
        """Advance ``tasks`` by one token.

        Args:
            tasks: Active tasks to advance.
            return_logprobs: Forwarded to the executor; per-token logprobs
                are recorded on each task's ``output_logprobs``.

        Returns:
            ``(decoded, aborted)``: tasks that produced a new token (its ID
            already appended to ``output_ids``) and tasks that hit the
            sequence cap and were marked ``ABORTED``.
        """
        to_prefill = [t for t in tasks if not t.prefill_done and t.prompt_ids]
        prefilled_ids = set()
        produced: List[Task] = []
        if to_prefill:
            for t in to_prefill:
                t.input_tokens = len(t.prompt_ids)

            groups: Dict[Tuple[int, Optional[AttentionBackend]], List[Task]] = {}
            for t in to_prefill:
                start_pos = min(
                    self._task_cache.task_cached(t.task_id), len(t.prompt_ids) - 1
                )
                groups.setdefault((start_pos, t.backend), []).append(t)

            for (start_pos, _), group in groups.items():
                backend = group[0].backend
                backend_context = (
                    attn_backend(backend) if backend is not None else nullcontext()
                )
                with (
                    backend_context,
                    self._metrics.record([t.task_id for t in group], "prefill"),
                ):
                    prefilled, pending = self._executor.execute_prefill(
                        group, start_pos=start_pos, return_logprobs=return_logprobs
                    )

                for t in prefilled:
                    t.mark_prefill_done()
                self.step_commit(pending)
                prefilled_ids.update(t.task_id for t in prefilled)
                produced.extend(prefilled)

                start_logical_page = start_pos // self._pool.page_size
                for t in group:
                    self._task_cache.task_record_hashes(
                        t.task_id, t.prompt_ids, start_logical_page
                    )

        decoded: List[Task] = []
        aborted: List[Task] = []
        if prefilled_ids:
            for t in tasks:
                if t.task_id in prefilled_ids:
                    continue
                if self._task_cache.task_extend(t.task_id, t.next_pos):
                    decoded.append(t)
                else:
                    t.status = TaskStatus.ABORTED
                    aborted.append(t)
        else:
            decoded, aborted = self._extend_and_partition(tasks)

        pending, produced_decoded, aborted_decoded = self._submit_decoded(
            decoded, return_logprobs, abort_on_refusal=True
        )
        aborted.extend(aborted_decoded)
        if pending is not None:
            self.step_commit(pending)
        produced.extend(produced_decoded)

        return produced, aborted

    def _extend_and_partition(self, tasks: List[Task]) -> Tuple[List[Task], List[Task]]:
        """Extend a step's tasks with ONE batched cache call.

        The steady decode step (no prefills in the batch) is the hot
        path: per-task ``task_extend`` costs one allocator round-trip
        each (~13us per task on serving-scale pools), the batched call
        harvests all new pages in one word-indexed pass.  Failure marks
        the task ABORTED exactly like the per-task loop.
        """
        ok = self._task_cache.task_extend_batch(
            [t.task_id for t in tasks], [t.next_pos for t in tasks]
        )
        decoded: List[Task] = []
        aborted: List[Task] = []
        for t, extended in zip(tasks, ok):
            if extended:
                decoded.append(t)
            else:
                t.status = TaskStatus.ABORTED
                aborted.append(t)
        return decoded, aborted

    def _submit_decoded(
        self, decoded: List[Task], return_logprobs: bool, abort_on_refusal: bool
    ) -> Tuple[Optional[PendingStep], List[Task], List[Task]]:
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
        produced: List[Task] = []
        aborted: List[Task] = []
        pending: Optional[PendingStep] = None
        for backend, group in self._task_backend_groups(decoded):
            backend_context = (
                attn_backend(backend) if backend is not None else nullcontext()
            )
            with (
                backend_context,
                self._metrics.record([t.task_id for t in group], "decode"),
            ):
                pending = self._executor.submit_decode(group, return_logprobs)
            if pending is None:
                # Refused combination (frequency penalty with an uncommitted
                # prior step): drain that step, then retry once.
                self._executor.flush_pending(self)
                with (
                    backend_context,
                    self._metrics.record([t.task_id for t in group], "decode"),
                ):
                    pending = self._executor.submit_decode(group, return_logprobs)
                if pending is None:
                    if abort_on_refusal:
                        for t in group:
                            t.status = TaskStatus.ABORTED
                            aborted.append(t)
                    continue
            for t in group:
                t.advance_kv()
            produced.extend(group)
        return pending, produced, aborted

    def step_submit(
        self, tasks: List[Task], return_logprobs: bool = False
    ) -> Tuple[List[Task], Optional[PendingStep]]:
        """Submit one step's work without committing results.

        The overlap scheduler's entry: prefill groups run through their
        usual submit path (their pending step is returned alongside the
        decode one — a batch mixing both yields at most one pending each
        per backend group; the last one wins and earlier ones must have
        been committed by ``flush_pending`` semantics).  Callers own the
        returned pending step and MUST commit it (or drain it via the
        executor) before touching the tasks again.
        """
        to_prefill = [t for t in tasks if not t.prefill_done and t.prompt_ids]
        prefilled_ids = set()
        produced: List[Task] = []
        pendings: List[PendingStep] = []
        if to_prefill:
            for t in to_prefill:
                t.input_tokens = len(t.prompt_ids)

            groups: Dict[Tuple[int, Optional[AttentionBackend]], List[Task]] = {}
            for t in to_prefill:
                start_pos = min(
                    self._task_cache.task_cached(t.task_id), len(t.prompt_ids) - 1
                )
                groups.setdefault((start_pos, t.backend), []).append(t)

            for (start_pos, _), group in groups.items():
                backend = group[0].backend
                backend_context = (
                    attn_backend(backend) if backend is not None else nullcontext()
                )
                with (
                    backend_context,
                    self._metrics.record([t.task_id for t in group], "prefill"),
                ):
                    prefilled, pending = self._executor.execute_prefill(
                        group, start_pos=start_pos, return_logprobs=return_logprobs
                    )
                if pending is not None:
                    pending.prefill_task_ids = tuple(t.task_id for t in prefilled)
                    pendings.append(pending)
                prefilled_ids.update(t.task_id for t in prefilled)
                produced.extend(prefilled)

                start_logical_page = start_pos // self._pool.page_size
                for t in group:
                    self._task_cache.task_record_hashes(
                        t.task_id, t.prompt_ids, start_logical_page
                    )

        decoded: List[Task] = []
        if prefilled_ids:
            for t in tasks:
                if t.task_id in prefilled_ids:
                    continue
                if self._task_cache.task_extend(t.task_id, t.next_pos):
                    decoded.append(t)
        else:
            decoded, _aborted = self._extend_and_partition(tasks)

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

    def step_commit(self, pending: PendingStep) -> List[Task]:
        """Materialise a submitted step onto its tasks.

        The single commit entry point: appends each sampled token (and its
        logprob, when recorded) to the task's output state and advances the
        KV write position.  Idempotent — a step already committed is a
        no-op — so abort paths can call it defensively.  Prefill callers
        mark ``prefill_done`` before this (the first token is already in
        the task's history at that point).

        A task that already reached its ``max_tokens`` when this step was
        in flight drops the extra token: the overlap pipeline may launch
        one step past the terminal one, and user-visible output must never
        run past the termination point.
        """
        if pending.committed:
            return []
        payload = pending.commit()
        produced: List[Task] = []
        preflip = pending.prefill_task_ids
        for task, (token_id, logprob) in zip(pending.tasks, payload):
            if task.task_id in preflip:
                task.mark_prefill_done()
            if task.max_tokens is not None and task.output_tokens >= task.max_tokens:
                continue
            task.output_ids.append(token_id)
            task.output_tokens += 1
            if logprob is not None:
                task.output_logprobs.append(logprob)
            produced.append(task)
        return produced
