# Inference

## Contents

- [KV Cache](#kv-cache)
- [KVCache System](#kvcache-system)
- [Attention Backend](#attention-backend)
- [Continuous Batching](#continuous-batching)
- [Sampling](#sampling-strategy-pattern)
- [Protocol Handlers](#protocol-handlers-strategy-pattern)
- [Engine & GenerateResult](#engine--generateresult)
- [HTTP API](#http-api) — endpoints, SSE, errors, stats
- [Engine API](#engine-api)

## KV Cache

At decode time, only the last query token matters. All previous K/V are cached to avoid recomputation:

$$
o_n = \sum_j \text{softmax}\left(\frac{q_n k_j}{\sqrt{d_k}}\right) v_j
$$

RoPE is applied **before** KV cache write, not after — otherwise position encoding drift occurs.

## KVCache System

Three-layer separation (SGLang-inspired): storage, index table, allocator.

```
PagePool (top-level manager, orchestrates all layers)
  ├── KVStorage              k_buffer / v_buffer [n_layers, size, n_kv_heads, head_dim]
  ├── ReqToTokenPool         req_to_token [num_reqs, max_ctx_len] → physical token slot
  ├── Allocator              bitmask-based page allocator + ref-count + LRU (paged mode only)
  └── PrefixCache            hash-based prefix matching (paged mode only)
```

`PagePool` supports two modes:

- **Contiguous (default)**: pre-allocates `max_batch_size * max_seq_len` token slots. `req_to_token` is a trivial linear mapping (`slot = req_idx * max_seq_len + pos`). No dynamic allocation.
- **Paged** (`page_size=1` or `>1` with `n_tokens` set): shared token pool with on-demand allocation. Allocator + PrefixCache enable prefix sharing and LRU eviction.

`bind_tasks()` returns a `KVCache` dataclass — pure data, no methods:

```
KVCache
  ├── k_buffer, v_buffer     [n_layers, size, n_kv_heads, head_dim]
  ├── req_to_token           [num_reqs, max_ctx_len]
  ├── req_pool_indices       [batch_size]
  ├── seq_lens               [batch_size]
  ├── out_cache_loc          [batch, seq_len] — write indices for this forward
  ├── max_len                int — max(seq_lens), avoids GPU sync in decode
  ├── kv_indptr              [batch + 1] int32 — prefix sum of seq_lens, precomputed once per step
  └── qo_indptr              [batch + 1] int32 — prefix sum of per-request q_lens (prefill), precomputed once per step
```

Attention layers do raw buffer indexing: `k_buffer[layer_id, out_cache_loc] = k` to write, `k_buffer[layer_id, indices]` to gather.

## Attention Backend

Attention computation (cache I/O + SDPA/kernel dispatch) is decoupled from the model via `AttentionBackend` ABC:

```
AttentionBackend (ABC)
  ├── TorchNativeBackend   SDPA + indirect KV cache gather (default)
  └── CudaBackend          CUDA kernel dispatch (attn_paged_decode, attn_paged_prefill)
```

Select via context manager (mirrors `torch.nn.attention.sdpa_kernel`):

```python
from astrai.extension import attn_backend, ATTN_BACKEND

with attn_backend(ATTN_BACKEND.CUDA):
    engine.generate("hello")
```

`CudaBackend` decode path: writes K/V to cache, then calls `attn_paged_decode` with `page_size=1` — the `req_to_token` table serves directly as the page table, each token slot is a single-token "page". No explicit K/V gather needed.

`CudaBackend` prefill path: writes K/V, then calls `attn_paged_prefill` — a ragged-batch (paged) prefill kernel that reads K/V directly from the flat pool via `req_to_token`, addressing each request's `q_len`/`kv_len` through `qo_indptr` and `kv_indptr`. No explicit K/V gather needed.

Fallback: `CudaBackend` delegates to `TorchNativeBackend` when a CUDA kernel is not available.

### Rotary Embedding Backend

Rotary embedding is applied via `apply_rotary_emb` in `astrai/extension/rotary_backend.py`, which auto-dispatches:

- **CUDA kernel** (`rotary_emb.cu`): fused cos/sin lookup + rotation in a single kernel, used when the kernel is available, the input is bf16 on CUDA, and `torch.is_grad_enabled()` is `False` (inference mode)
- **Torch fallback**: complex multiply path (`torch.view_as_complex` → `torch.complex` multiply → `torch.view_as_real`), used during training (supports autograd backward) or when the CUDA kernel is not available

`RotaryEmbedding` stores a cos/sin table `freqs_cis` of shape
`[max_len, dim/2, 2]` (f32 — `[cos, sin]` pairs) and `forward()` returns
a `[batch, seq_len, dim/2, 2]` slice indexed by `position_ids`. Both
attention backends share the same rotary dispatch — it is backend-agnostic.

## Continuous Batching

`InferenceScheduler` runs a daemon thread with a 4-phase loop:

```
1. Cleanup → Remove finished tasks, free KV cache slots/pages
2. Refill  → Pop from waiting_queue, task_alloc resources, activate
3. Prefill → Group by (prompt_len, start_pos), run full forward
4. Decode  → Run single-token forward for each same-position group
```

## Sampling (Strategy Pattern)

```
BaseSamplingStrategy (ABC)
  ├── TemperatureStrategy
  ├── TopKStrategy
  ├── TopPStrategy
  └── SamplingPipeline
```

`SamplingPipeline` composes them: Temperature → Top-K → Top-P → softmax → multinomial.  
`sample()` is a convenience shortcut for one-shot usage.

## Protocol Handlers (Strategy Pattern)

```python
class ProtocolHandler:  # concrete orchestrator
    def __init__(self, request, engine, builder): ...
    async def handle(self):
        prompt, ctx, stops = builder.prepare(request, engine)
        agen = engine.generate_async(prompt, ...)
        if stream: self._handle_stream(agen, ctx, stops)
        else:      return await self._handle_non_stream(agen, ctx, stops)
```

`ResponseBuilder` (ABC): `prepare()`, `format_stream_start()`, `format_chunk()`, `format_stream_end()`, `format_response()`.

`OpenAIResponseBuilder` → `/v1/chat/completions`, `AnthropicResponseBuilder` → `/v1/messages`.

Adding a protocol = one builder file, no handler subclassing needed.

## Engine & GenerateResult

```
InferenceEngine
  ├── generate(prompt, stream, ...) → str | List[str] | Generator
  ├── generate_with_request(req)    → same
  ├── generate_async(prompt, ...)   → AsyncGenerator
  ├── get_stats()                   → Dict
  └── shutdown()
```

`GenerateResult` uses `Condition` for non-streaming (`wait_completion()`) and `Event` for streaming (`wait()`). Stream callback is `cb(token)`.

## HTTP API

```
POST /v1/chat/completions   OpenAI
POST /v1/messages            Anthropic
GET  /health                 {"status":"ok","model_loaded":true}
GET  /stats                  scheduler statistics
```

### OpenAI

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

Response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1717000000,
  "model": "astrai",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
}
```

Streaming SSE: `object: "chat.completion.chunk"` — starts with role delta, then token chunks, ends with finish chunk + usage stats, then `data: [DONE]`.

### Anthropic

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"astrai","system":"You are helpful.","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

Supports `stop_sequences` and streaming via `event: content_block_delta`. Anthropic streams also end with the shared `data: [DONE]` sentinel after `event: message_stop`.

### Request Parameters

The HTTP protocols and direct engine API have distinct request models and defaults.

**OpenAI** (`ChatCompletionRequest`):

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | `"astrai"` | Model name returned in responses |
| `messages` | List[dict] | required | Chat messages (role, content) |
| `temperature` | Optional[float] | 1.0 | Sampling temperature (0.0-2.0) |
| `top_p` | Optional[float] | 1.0 | Nucleus threshold (0.0-1.0) |
| `top_k` | Optional[int] | 50 | Top-k count |
| `max_tokens` | Optional[int] | 2048 | Max generation length |
| `stream` | Optional[bool] | False | Stream output |
| `stop` | Optional[Union[str, List[str]]] | None | Stop sequences |
| `n` | Optional[int] | 1 | Number of choices requested |
| `presence_penalty` | Optional[float] | 0.0 | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | Optional[float] | 0.0 | Frequency penalty (-2.0 to 2.0) |
| `logit_bias` | Optional[Dict[int, float]] | None | Per-token logit bias |
| `user` | Optional[str] | None | End-user identifier |
| `tools` | Optional[List[ToolDef]] | None | Tool definitions for function calling |
| `tool_choice` | Optional[Union[str, Dict[str, Any]]] | `"auto"` | Tool selection mode or explicit tool choice |

**Anthropic** (`MessagesRequest`):

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | `"astrai"` | Model name returned in responses |
| `messages` | List[AnthropicMessage] | required | User/assistant messages |
| `system` | Optional[str] | None | System prompt |
| `max_tokens` | int | 1024 | Max generation length |
| `temperature` | Optional[float] | 1.0 | Sampling temperature (0.0-2.0) |
| `top_p` | Optional[float] | 1.0 | Nucleus threshold (0.0-1.0) |
| `top_k` | Optional[int] | 50 | Top-k count |
| `stream` | Optional[bool] | False | Stream output |
| `stop_sequences` | Optional[List[str]] | None | Stop sequences |

**Engine** (`GenerationRequest`):

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | List[Dict[str, str]] | required | Messages to format before generation |
| `top_k` | int | 50 | Top-k count; 0 disables filtering |
| `top_p` | float | 1.0 | Nucleus threshold |
| `temperature` | float | 1.0 | Sampling temperature; 0 enables greedy decoding |
| `max_tokens` | Optional[int] | None | Max generation length |
| `frequency_penalty` | float | 0.0 | Frequency penalty (-2.0 to 2.0) |
| `rep_window` | int | 64 | Recent-token window used by the frequency penalty |
| `stream` | bool | False | Stream output |

### SSE Streaming Format

**OpenAI** (`/v1/chat/completions`, `stream=true`):

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"astrai",
       "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":0,"model":"astrai",
       "choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"astrai",
       "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}

data: [DONE]
```

**Anthropic** (`/v1/messages`, `stream=true`):

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","model":"astrai","role":"assistant",
       "content":[],"usage":{"input_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{...}}

event: message_stop
data: {"type":"message_stop"}

data: [DONE]
```

### Error Responses

The server returns standard HTTP status codes. Pydantic validation errors (e.g. missing required fields)
are handled automatically by FastAPI with 422 status. The only application-level error is engine initialization:

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 422 | Unprocessable entity (Pydantic validation) |
| 503 | Service unavailable (model not loaded, engine not ready) |

Error response body (503):

```json
{
    "detail": "Engine not initialized"
}
```

### Stats Endpoint

```
GET /stats
```

Response:

```json
{
    "total_tasks": 128,
    "total_tokens": 10240,
    "active_tasks": 3,
    "waiting_queue": 2
}
```

## Engine API

```python
# Non-streaming
engine.generate("Hello", stream=False)          # -> str
engine.generate(["A", "B"], stream=False)       # -> List[str]

# Streaming
engine.generate("Hello", stream=True)           # -> Generator[str]
engine.generate(["A", "B"], stream=True)        # -> Generator[Tuple[int, str]]

# Async
async for token in engine.generate_async("Hello", ...):    # -> AsyncGenerator[str]
    print(token)
```

> Document Update Time: 2026-07-31
