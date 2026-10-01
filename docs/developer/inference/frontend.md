# Frontend Layer

`astrai/inference/frontend/` — everything that faces the user. The vLLM
analogue is the API process (`AsyncLLM` / `LLMEngine` +
`InputProcessor` / `OutputProcessor`); here it is an in-process layer so
colocated RL rollout keeps sharing the model object with the trainer.

Frontend code must not touch engine locks, hold `Request` objects beyond
submission, or read KV state.

## Classes

```mermaid
classDiagram
    direction TB
    class InferenceEngine {
        <<FACADE>>
        +nn.Module model
        +AutoTokenizer tokenizer
        -EngineCoreClient _core
        -RequestTracker _tracker
        -InputProcessor _input_processor
        +generate(prompt, stream, max_tokens, ...) Union~Generator, str, List~str~~
        +generate_async(prompt, ...) AsyncGenerator~str~
        +generate_events(prompt, ..., stop_sequences) AsyncGenerator~StreamChunk~
        +score(prompt, continuation, per_token)
        +get_stats() Dict
        +shutdown()
    }
    class EngineCoreClient {
        <<abstract · STRATEGY>>
        +send_request(**kwargs) str
        +send_requests(prompts, **kwargs) List~str~
        +abort_request(request_id) bool
        +stats() Dict
        +shutdown()
    }
    class InprocClient {
        T0: direct method calls
        T1 swaps in a transport client
    }
    class InputProcessor {
        +AutoTokenizer _tokenizer
        +int _max_seq_len
        +new_request_id() str
        +process(prompt) ProcessedInput
        +process_batch(prompts) List~ProcessedInput~
    }
    class ProcessedInput {
        <<frozen>>
        +str request_id
        +List~int~ prompt_ids
    }
    class OutputProcessor {
        +ProcessedOutput state
        +push(event) Tuple
        +usage() Tuple~int, int~
        +finished bool
    }
    class StopSequenceChecker {
        +Optional~str~ matched
        +push(text) Tuple~str, bool~
        incremental windowed matching
    }
    class RequestTracker {
        +EventQueueSink sink
        +register(request_id, maxlen) Event
        +drain(request_id) List
        +wait(request_id, timeout) bool
        +is_finished(request_id) bool
    }
    class GenerateResult {
        public accumulator for streaming adapters
        +tokens / results
        +append_batch(items)
        +wait_completion(timeout)
        +get_results() List~str~
    }
    class OutputEvent {
        <<frozen · core/events.py>>
        TokenDelta(request_id, token_id, sequence_no)
        RequestFinished(request_id, finish_reason, usage)
        RequestError(request_id, code, message, retryable)
    }
    class StreamChunk {
        +text
        +delta_token_ids / current_token_ids
        +finish_reason / prompt_tokens / completion_tokens
    }

    InferenceEngine *-- EngineCoreClient
    EngineCoreClient <|.. InprocClient
    InferenceEngine --> InputProcessor : uses
    InferenceEngine --> OutputProcessor : folds events
    InferenceEngine *-- RequestTracker
    InputProcessor ..> ProcessedInput : produces
    OutputProcessor *-- StopSequenceChecker
    RequestTracker ..> OutputEvent : consumes
    OutputProcessor ..> StreamChunk : frontend maps
```

## Responsibilities

- **`InputProcessor`** tokenizes on the caller's thread (single or one
  batched `encode` call) and **mints the request id before submission**.
  Because the id exists before the scheduler knows about the request, an
  event can never precede its consumer — this is what allowed the old
  `_ResultSink` replay buffer to be deleted.
- **`OutputProcessor`** folds events for one request: incremental
  detokenization (via `StreamDecoder`), stop-sequence matching with a
  sliding window over the ambiguous tail (never a full-body substring
  scan), and exact usage accounting from token deltas (never re-encoding
  text to count tokens). Degrades to token-id-as-string when the
  tokenizer lacks the Rust streaming handle, so a stream always
  terminates.
- **`EngineCoreClient`** is the only path from frontend to core. T0 uses
  `InprocClient` (plain method calls on the live `Scheduler`); a T1
  serving deployment replaces it with a transport client and nothing
  else in the engine or the protocol adapters changes.
- **`RequestTracker`** holds bounded per-request event queues fed by the
  scheduler loop thread. The loop thread only appends; all consumer work
  (detokenize, protocol formatting, user callbacks) runs where the queue
  is drained. The sink flags the request's finished ``Event`` the moment
  it queues the terminal event — completion is observable without a
  consumer folding the stream.

## Blocking generate is completion-driven

Non-streaming `generate` runs **no helper thread**. It submits, parks the
caller on the per-request finished events, and when every request has
terminated folds all events at once on the caller's thread
(`_collect_blocking`). Per-step fold wake-ups were a measured host
overhead: each wake is a GIL handoff with the scheduler loop at exactly
the worst time (between commit and the next submit). Folding once, after
the batch is done, moves the same work off the decode loop entirely; the
event queues for blocking generate are sized to `max_seq_len + 1` so
accumulating until completion never drops tokens. Streaming entry points
keep incremental folding — their consumers want text as it is produced.

## Error containment

The completion-driven fold is fault-tolerant: a per-request fold failure
keeps that request's partial text instead of losing the whole batch, and
a `RequestError` terminal event is logged and reported through the same
partial-text path rather than hanging the caller. The scheduler's
`_emit_events` catches sink exceptions — consumer failures never reach
the engine loop.
