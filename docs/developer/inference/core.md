# Core Layer

`astrai/inference/core/` — the engine core (vLLM analogue: the
EngineCore process's scheduler + KV accounting; here a single-writer
loop thread in-process). The core emits output events; it never
detokenizes and never runs consumer callbacks.

## Classes

```mermaid
classDiagram
    direction TB
    class Scheduler {
        +run_busy_loop()
        +run_batch(prompt_ids_list, ...) List
        +score_ids(prompts, conts, per_token) List
        +add_request(prompt, **kwargs) str
        +add_requests(prompts, **kwargs) List~str~
        +cancel_request(request_id) bool
        +set_event_sink(sink)
        +start() / stop()
        +update_weights(policy_version) int
        +apply_weight_update(policy_version, update)
        single-writer loop thread:
        cleanup → admit → overlap step → emit events
    }
    class OutputEventSink {
        <<abstract · observer outlet>>
        +__call__(events)
        invoked on the loop thread; must be fast
    }
    class CallbackBridge {
        default sink: events → legacy
        stream-callback protocol (STOP sentinel)
    }
    class RequestManager {
        +AutoTokenizer tokenizer
        +Deque waiting
        +List running
        +add_request(s)(prompt, ..., request_id, prompt_ids)
        +cancel_request(request_id) Tuple
        +remove_finished_requests(stop_ids) List~Request~
        +pull_waiting(n) / activate(request)
        +invoke_callbacks(events)
        +get_running_requests() / get_waiting_requests()
    }
    class Request {
        +str request_id
        +List prompt_ids
        +List output_ids
        +int num_computed_tokens
        +int input_tokens / output_tokens
        +RequestStatus status
        +mark_prefill_complete()
        +advance_kv(n)
        +next_pos int
        +is_finished(stop_ids) bool
    }
    class RequestStatus {
        <<enumeration>>
        PENDING
        RUNNING
        FINISHED
        ABORTED
    }
    class SchedulerStep {
        +step(requests) Tuple
        +step_submit(requests) Tuple
        +step_commit(pending) List~Request~
        decode-first → continuation chunk → new first chunk
    }
    class PolicyVersionGuard {
        +RLock lock
        +int policy_version
        +update_weights(version) int
        +apply_weight_update(version, update)
        +with_policy_snapshot(inspect)
        monotonic version over shared weights
    }
    class MetricsCollector {
        +register(request_id)
        +mark_finished(request_id, in, out)
        +record(request_ids, phase)
    }

    class KVCacheManager {
        +Dict states request_id → RequestCacheState
        +alloc_slots(request_id, prompt_ids) bool
        +free_slots(request_id)
        +extend_slots(request_id, pos) bool
        +extend_slots_batch(ids, positions) List~bool~
        +cached_tokens(request_id) int
        +record_block_hashes(request_id, ids, start)
        +bind(request_ids, workspace, start_pos) KVCache
        +invalidate_cache() int
    }
    class BlockPool {
        +KVStorage storage
        +ReqToTokenPool req_pool
        +int page_size
        +bind_tasks(...) KVCache
    }
    class AllocationStrategy {
        <<abstract · STRATEGY>>
        +alloc(state, prompt_ids) bool
        +free(state)
        +extend(state, pos) bool
        +extend_batch(states, positions)
    }
    class ContiguousStrategy {
        static partition, no prefix cache
    }
    class PagedStrategy {
        +Allocator alloc bitmap+LRU
        +RadixCache prefix
        extend_batch: one word-indexed harvest
    }
    class RequestCacheState {
        +int req_idx
        +int length / cached
        +List pages
        +List slots host-staged tail
    }

    Scheduler *-- RequestManager
    Scheduler *-- SchedulerStep
    Scheduler *-- KVCacheManager
    Scheduler *-- PolicyVersionGuard
    Scheduler *-- MetricsCollector
    Scheduler --> OutputEventSink : emits via _emit_events
    OutputEventSink <|.. CallbackBridge
    RequestManager o-- Request : waiting / running
    KVCacheManager --> BlockPool : delegates
    BlockPool *-- AllocationStrategy
    AllocationStrategy <|.. ContiguousStrategy
    AllocationStrategy <|.. PagedStrategy
    KVCacheManager o-- RequestCacheState : per-request
```

## The busy loop

`Scheduler.run_busy_loop` is a plain Python loop on its own thread (no
asyncio — the vLLM EngineCore discipline). Each iteration:

1. **cleanup** — `remove_finished_requests(stop_ids)`; a finished
   request whose KV is still written by the in-flight step is *retired*
   (slot freed next iteration, after the pending step commits);
2. **admit** — waiting requests get KV slots (`alloc_slots`);
3. **step** — steady decode batches ride the depth-2 submit/commit
   overlap pipeline (`enable_overlap`); any batch change drains first
   and falls back to the synchronous step;
4. **emit** — one `OutputEvent` batch per step: `TokenDelta` per
   committed token plus `RequestFinished`; no text, no callbacks.

## KV accounting

`KVCacheManager` is the facade the scheduler talks to; `BlockPool` owns
the physical buffers (`KVStorage` + `ReqToTokenPool`) and an
`AllocationStrategy`:

- **ContiguousStrategy** — static per-request partitions (the default
  constructor path; no prefix cache).
- **PagedStrategy** — dynamic pages from a bitmask allocator (word-index
  harvest for batched extends), optional `RadixCache` prefix index keyed
  by exact token-page edges. Only complete, materialized pages are
  shared; the final sampled token is excluded (not yet in KV).

`Request.num_computed_tokens` is the single scheduling state variable —
the vLLM model: every optimization (chunked prefill, prefix hits,
overlap) changes how it advances, not the scheduler's branches.

## Policy versioning

`PolicyVersionGuard` is the train-serve weight protocol (no vLLM
counterpart): weights are mutated in place under its RLock; versions are
monotonic; every commit invalidates cached KV produced by older weights
(`invalidate_cache`). Weight updates require the loop stopped and queues
drained (`_ensure_weight_update_ready`).
