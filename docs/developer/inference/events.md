# Event Flow and Request Lifecycle

The event contract (`core/events.py`) is the only channel the scheduler
loop uses to report progress. This page shows the flow end-to-end and
the state machines it drives.

## One online generation, end to end

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller / HTTP
    participant E as InferenceEngine (frontend)
    participant IP as InputProcessor
    participant T as RequestTracker
    participant S as Scheduler (core · loop thread)
    participant R as GPUModelRunner (worker)
    participant OP as OutputProcessor

    C->>E: generate_events(prompt, stop_sequences)
    E->>IP: process(prompt)
    IP-->>E: ProcessedInput(request_id, prompt_ids)
    E->>T: register(request_id)
    E->>S: send_request(...) via InprocClient
    loop every loop iteration
        S->>S: cleanup → admit → step (budget)
        S->>R: submit_decode (no host resolve)
        R-->>S: PendingExecution (device tokens)
        S->>S: step_commit → state advance
        S->>T: sink([TokenDelta, …])
    end
    T-->>OP: drain(request_id)
    OP->>OP: detokenize · stop window · usage
    OP-->>C: StreamChunk(text, delta_ids, …)
    S->>T: sink(RequestFinished(reason, usage))
    OP-->>C: final StreamChunk(finish_reason)
    E->>T: unregister(request_id)
```

Key property: the loop thread only ever **appends events to queues**
(`_EventQueueSink.__call__` is a bucket-append under one lock). All
consumer work — detokenization, stop matching, protocol formatting, user
callbacks — runs where the queue is drained. A sink exception is caught
by `_emit_events` and logged; the loop keeps running.

## Request state machine

```mermaid
stateDiagram-v2
    [*] --> WAITING : add_request (frontend minted id, KV admitted)
    WAITING --> RUNNING : activate (alloc_slots succeeded)
    WAITING --> WAITING : alloc refused → back to queue
    RUNNING --> RUNNING : num_computed_tokens += chunk / step
    RUNNING --> FINISHED : stop id · max_tokens · length
    RUNNING --> ABORTED : KV extension failure · cancel
    WAITING --> ABORTED : cancel while queued
    FINISHED --> [*]
    ABORTED --> [*]
```

`num_computed_tokens` is the single scheduling state variable (the vLLM
model). Prefill chunks advance it by the chunk length (only the final
chunk samples); decode steps advance it by one. It always runs ahead of
`output_tokens` by at most one in-flight step — that gap is the
optimistic-advance window, and `step_commit` is its only correction
point.

## Event taxonomy

| Event | Meaning | Terminal |
|-------|---------|----------|
| `TokenDelta(request_id, token_id, sequence_no)` | one committed output token | no |
| `RequestFinished(request_id, finish_reason, prompt_tokens, completion_tokens)` | success termination | yes |
| `RequestError(request_id, error_code, message, retryable)` | failure termination | yes |

Finish reasons: `stop_token`, `length`, `cancelled`, `aborted`,
`rejected` — protocol-neutral; the OpenAI/Anthropic adapters map them
onto their own vocabularies. Every request ends with exactly one
terminal event, emitted after all of its `TokenDelta` events.

## Legacy bridge

Consumers that registered plain or batched stream callbacks (the
pre-event protocol) are served by `_CallbackBridge`, the default sink:
it detokenizes per request (sink side, not the loop's emission code) and
delivers `(request_id, text)` batches, terminal events becoming the
`STOP` sentinel. `run_batch`/`score` bypass the event stream entirely —
they drive the executor on the calling thread for RL rollout.

## Design-pattern view

| Pattern | Where | Why |
|---------|-------|-----|
| Facade | `InferenceEngine`, `KVCacheManager` | one entry surface each |
| Strategy | `EngineCoreClient`/`InprocClient`, `AllocationStrategy`, sampling chain | swap deployment/allocation/sampling without touching callers |
| Observer (push) | `OutputEventSink` outlet, queue sink, callback bridge | consumers off the producer thread; T1 swaps the sink for a transport |
| Command | `PendingExecution` submit/commit | deferred, idempotent execution; the seam for overlap |
| Flyweight | `InferenceWorkspace` | fixed-address buffers, CUDA-graph capture |
| Guard | `PolicyVersionGuard` | weight publication protocol |
| Memento | `RequestCacheState` | allocation snapshot, rollback on failed admission |

Deliberately absent: plugin registries/abstract factories (30-backend
scale not reached), mixin towers (SGLang's lesson), and a process
boundary at the engine core (would break `PolicyVersionGuard`'s shared
model object).
