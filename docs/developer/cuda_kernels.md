# CUDA Kernels

AstrAI includes optional custom CUDA kernels for attention, rotary embedding,
BF16 GEMV/SwiGLU, and FP8 GEMM. These are built when `nvcc` is available and
CUDA is detected. BF16 GEMV and SwiGLU are directly callable and can be
selected by guarded model dispatchers described below.

## Overview

| Kernel | File | Description |
|--------|------|-------------|
| `attn_decode` | `attention/decode.cu` | GQA decode attention (split-KV) |
| `attn_prefill` | `attention/prefill.cu` | GQA prefill attention (split-Q) |
| `attn_paged_decode` | `attention/paged_decode.cu` | Paged KV cache decode attention |
| `attn_paged_prefill` | `attention/paged_prefill.cu` | Paged KV cache prefill attention (ragged batch) |
| `rotary_emb` | `rotary_emb.cu` | Fused rotary embedding (cos/sin lookup + rotation) |
| `bf16_gemv` | `gemv/bf16_gemv.cu` | M=1..8 BF16 linear with FP32 accumulation (sm_80+) |
| `bf16_swiglu` | `gemv/bf16_swiglu.cu` | Fused M=1..8 BF16 up/gate projections and SwiGLU epilogue (sm_80+) |
| `fp8_ops` | `fp8/ops.cu` | FP8 quantization + tensor-core GEMM (sm_89+) |

### BF16 GEMV primitive

`astrai.extension.bf16_gemv(x, weight, bias=None)` accepts a contiguous BF16
input shaped `[K]` or `[M, K]`, with `M` in `[1, 8]` and any positive `K`, and
row-major weights `[N, K]`. The general path assigns one 256-thread CTA to an
output row and computes all M results together, reusing the weight row across
tokens. For measured aligned M=4 medium projections, a 128-thread CTA instead
assigns one output to each of four warps. That removes the CTA-wide reduction
barrier and exposes four neighboring outputs without changing accumulation.

The weight stream uses 128-bit vectorized loads anchored at each row's first
16-byte-aligned address with scalar head/tail sweeps for unaligned remainders,
so arbitrary `K` and storage offsets stay correct. The warp-tiled path is used
only when both tensors and every row are 16-byte aligned; all other calls keep
the general arbitrary-K path. Accumulation is FP32; the optional BF16 bias is
fused before the BF16 store. The launcher uses the current CUDA stream, is
CUDA Graph capture-safe, and requires sm_80 or newer.

Model `Linear` calls route through the lightweight linear backend. Set
`ASTRAI_GEMV=0` for an unconditional `F.linear` fallback, `1` to force the
kernel for any supported M in [1, 8], or `auto` (the default) to select only
architecture/shape bands that pass both the per-shape and end-to-end gates.
Measured SM89 small-M bands are enabled as follows:

| M | Automatic `(N, K)` bands | Validated gain |
|---:|---|---:|
| 1 | OPT-1.3B Q/K/V/O and MLP | +4.54% OPT projection chain |
| 2 | AstrAI `(256,1536)`, `(1536,1536)`, `(100000,1536)` plus all common shapes below | +14.0% on AstrAI 1B; +5.66% to +25.20% common chains |
| 4 | AstrAI `(256,1536)`, `(1536,1536)` plus gated common shapes below | +11.8% on AstrAI 1B; +5.67% to +7.71% common chains |
| 8 | none | at least one projection in every measured family missed the per-shape gate |

The common set covers LLaMA 2 7B Q/O, gate/up, and down; LLaMA 3 8B K/V,
gate/up, and down; LLaMA 2 13B Q/K/V/O, gate/up, and down; and GPT-NeoX MLP
up/down. In `(N,K)` form it is `(1024,4096)`, `(4096,4096)`,
`(11008,4096)`, `(4096,11008)`, `(14336,4096)`, `(4096,14336)`,
`(5120,5120)`, `(13824,5120)`, `(5120,13824)`, `(16384,4096)`, and
`(4096,16384)`. M=2 enables all eleven. M=4 excludes the three LLaMA 2 7B
bands `(4096,4096)`, `(11008,4096)`, and `(4096,11008)` because their combined
projection chain reached only +1.89%, below the 3% automatic-dispatch gate.

The extended common set adds Qwen2-7B `(512,3584)`, `(3584,3584)`,
`(18944,3584)`, and `(3584,18944)`; LLaMA 3 70B `(1024,8192)`,
`(8192,8192)`, `(28672,8192)`, and `(8192,28672)`; and OPT-1.3B
`(2048,2048)`, `(8192,2048)`, and `(2048,8192)`. Qwen2 and LLaMA 3 70B are
enabled at M=2/4. OPT-1.3B is enabled at M=1/2. Other rows retain their
previous policy or fall back to PyTorch.

Inside the primitive, a templated cooperative kernel uses either 256 threads
or a shape-gated 128-thread CTA. The smaller CTA is enabled only where an
interleaved direct-module comparison against the original 256-thread kernel
cleared 5%: OPT up at M=1; selected LLaMA 2 7B, Qwen2, and OPT projections at
M=2; LLaMA 2 13B Q/O, Qwen2 Q/O, and selected OPT projections at M=4; and
selected LLaMA 2, Qwen2, LLaMA 3 KV, and OPT projections at M=8. Confirmed
direct-kernel gains range from +5.37% to +48.54%. Long-K and saturated shapes
keep the 256-thread fallback. This internal selector is separate from model
automatic dispatch, whose Python/wrapper overhead is included in the gates
above.

On NVIDIA L20 (SM89), the common-shape microbenchmark reports +5.37% to
+114.39% for M=2 and +5.38% to +115.26% for M=4 versus `F.linear`. The paired
main-versus-warp-tiling run used identical interleaved settings; for the four
M=4 selected shapes, candidate latency changed from 0.016292 to 0.016108 ms
for `(4096,4096)`, 0.037939 to 0.028539 ms for `(11008,4096)`, 0.043407 to
0.039803 ms for `(4096,11008)`, and 0.006697 to 0.006390 ms for
`(1024,4096)`.

The dependent projection-chain gate, which includes Python dispatch and
rotates through distinct weights instead of repeatedly warming one matrix,
measured:

| Synthetic chain | M=2 | M=4 | Row argmax parity |
|---|---:|---:|---|
| LLaMA 2 7B | +8.49% | fallback (M=4 bands excluded) | exact |
| LLaMA 3 8B | +8.50% | +6.44% | exact |
| LLaMA 2 13B | +5.66% | +5.67% | exact |
| GPT-NeoX 20B | +6.95% | +5.93% | exact |
| Qwen2 7B | +7.48% | +7.48% | exact |
| LLaMA 3 70B | +7.77% | +7.69% | exact |
| OPT 1.3B | +25.20% | fallback (M=4 up projection regresses) | exact |

OPT 1.3B M=1 is +4.54%. Qwen2 and LLaMA 3 70B M=1, and all three new
families at M=8, remain exact PyTorch fallbacks.

These are synthetic projection-chain measurements, not whole-model throughput
claims. Reproduce them with `scripts/tools/benchmark_gemv_common.py`.

The AstrAI 1B A→B→B→A results use the real `InferenceEngine`, including scheduler,
sampling, and CUDA Graph. M=8 stays on PyTorch because its remaining
greedy-stable winners missed the 3% end-to-end gate. Long-K MLP-down bands are
also excluded because their valid BF16 error changed a checkpoint greedy
argmax; the enabled M=2/4 bands matched the baseline greedy output exactly.

Training, prefill, unmeasured architectures, and losing shape bands always
remain on PyTorch. Use mode `1` only for explicit A/B runs outside this table.

The primitive remains directly callable and deliberately has no internal
`F.linear` fallback. The model-level backend owns fallback and dispatch policy.

### BF16 SwiGLU primitive

`astrai.extension.bf16_swiglu(x, up_weight, gate_weight)` fuses the two dense
MLP projections with `up * silu(gate)` into one CUDA launch for contiguous
BF16 inputs with `M` in `[1, 8]` and K divisible by 8. It preserves the BF16
rounding boundaries of the two projection outputs, SiLU output, and final
product while accumulating dot products in FP32.

The kernel contains two output-row tilings. A CTA-reuse path reads each up/gate
weight chunk once and applies it to all M rows. The native AstrAI 1B shape
`(N,K)=(6912,1536)` uses one warp per decode row for M=2/4/8; on L20 this
removes the shared reductions and barrier and reduces M=4 CUDA-Graph latency
from 0.0324 ms to 0.0181 ms. Wider LLaMA/GPT-NeoX matrices keep CTA reuse,
because duplicating their weight reads across row warps regressed 1.3-4.2%.

Dense `MLP` modules route through the SwiGLU backend. `ASTRAI_SWIGLU=0` keeps
the unfused linear backend, and `1` explicitly forces the fused primitive.
`auto` is the default but currently has no enabled bands: although direct
errors are small (maximum absolute error at most 2.4e-4 in the L20 matrix),
the different FP32 reduction order changed greedy checkpoint output for
M=1/2/4. Automatic dispatch therefore remains numerically identical to the
existing path. See [the benchmark protocol](./swiglu_benchmark.md) for raw
operator, engine, and checkpoint evidence.

Additionally, optimized `.cuh` variants with tensor-core MMA (Matrix Multiply-Accumulate) exist:

| Variant | File | Optimization |
|---------|------|--------------|
| Split-KV MMA decode | `attention/decode_split_kv_mma.cuh` | Split KV across warps + MMA (sm_80+) |
| Split-Q MMA prefill | `attention/prefill_split_q_mma.cuh` | Split Q across warps + MMA (sm_80+) |

> The paged and non-paged paths share one kernel body. Prefill is templated on
> an independent Q schedule (`DenseQSchedule` / `PackedQSchedule`) and KV
> source (`ContigKV` / `PagedKV`); decode only needs the KV source. There are
> no separate `attn_paged_*.cuh` files.

### Rotary Embedding Kernel

The `rotary_emb` kernel (`csrc/kernels/rotary_emb.cu`) fuses cos/sin lookup and rotation into a single kernel:

- One thread per (head, dim-pair), vectorized `__nv_bfloat162` load/store
- f32 cos/sin input, bf16 compute and output
- 256-thread blocks, grid-stride loop
- Auto-dispatched via `apply_rotary_emb` in `astrai/extension/backend/rotary.py` (CUDA when available + inference mode, else torch complex-multiply fallback)
- No context-manager backend needed — rotary is backend-agnostic, both attention backends benefit

Standalone benchmark vs torch complex-multiply (48 calls = 24 layers × q+k): 6-9x faster, max diff 0 (decode) to 3e-2 (large prefill, bf16).

### FP8 GEMM / Linear Kernel

The `fp8_ops` family (`csrc/kernels/fp8/`) accelerates bf16 linear layers by
quantizing to FP8 and running tensor-core GEMMs (**requires sm_89+**; fp8
`mma.sync.m16n8k32` only exists on Ada/Hopper). Same three-layer style as
attention; the GEMM device code is split humming/CUTLASS-style into one
layered directory:

| File | Role |
|------|------|
| `fp8/common.h` | `FP8Format` enum (E4M3/E5M2), `Fp8GemmTraits<Fmt, BlockM, BlockN, K, Stages>`, `FP8Params` / `FP8QuantizeParams` PODs, layout tags — no torch |
| `fp8/quantize.cuh` | pure-CUDA device code: vectorized `fp8_quantize_kernel` + 32×32-tile transpose kernel (out_layout 0/1/2), `quant_in_traits<InT>` unpack — no torch |
| `fp8/gemm/policy.cuh` | smem budget / occupancy hint (`Fp8GemmSmem`) + `Fp8GemmPolicy` (traits + layouts + knobs — the kernel's single template parameter) |
| `fp8/gemm/load.cuh` | operand loaders: swizzle (`tile_at`), congruous cp.async (predicated + interior), `PrefetchCarry`, crosswise LDG+PRMT direct load |
| `fp8/gemm/scheduler.cuh` | CTA id → (block_m, block_n) grouped/plain raster |
| `fp8/gemm/mainloop.cuh` | `Fp8CollectiveMainloop`: stage rings, stage loads, fragment addressing, pipelined mma.sync loop |
| `fp8/gemm/epilogue.cuh` | `Fp8CollectiveEpilogue`: fused bias + bf16 smem scatter + coalesced copy-out |
| `fp8/gemm.cuh` | umbrella: `fp8_gemm_kernel<Policy>` orchestrator + host planning (`plan_gemm` / `launch_plan`; 64×64 / 128×64 / 128×128 CTA) + entry `gemm<Fmt>(params, stream, trans_a, trans_b)` = `canonicalize_gemm` → `plan_gemm` → `launch_plan` |
| `fp8/ops.cu` | binding only: `check_fp8_device` (sm_89+), param packing, launch dispatch, pybind → module `fp8_ops` |

Scale semantics: `quantize` takes the quantization *multiplier*; the
strategy layer passes `scale.reciprocal()` and the kernel multiplies by it.
`mm_fp8` takes the combined dequant scale (`sa * sb`). `amax` is always
returned in the original input domain.

Python layer (two levels): `astrai/extension/ops/fp8.py` provides stateless
primitives (`fp8_quantize` / `fp8_gemm`) via `torch.library.custom_op`, with
plain `quantize` / `mm_fp8` wrappers, and `astrai/extension/fp8.py` is the
strategy layer (`fp8_autocast`, delayed / dynamic scaling recipes,
`fp8_linear_forward/backward` wiring `aten::linear` on CUDA). See the FP8
section in `AGENTS.md` for full detail.

#### FP8 GEMM design notes

The load-bearing invariants behind the kernel code (all measurements on
L20/sm_89 unless noted):

**Swizzle.** Staging tiles are flat `[rows * kK]`; `tile_at` XORs the 16B
chunk index with row bits at `[3, 3+log2(kChunks))` so a warp's ldmatrix
fragment load (8 consecutive rows × 16B) hits all 32 banks exactly once
(the unswizzled row word-stride is `kK/4` words, so rows `r` and
`r + 8/kChunks` collide mod 32). Chunks stay contiguous, so cp.async
staging is unaffected.

**Fragment addressing (base-pair scheme).** One base register per operand
per k_seg, every fragment offset an LDSM immediate. The closure works
because the XOR swizzle's source bits come only from the lane's
row-within-matrix `r7`: the 8/16-row fragment steps never reach them, so
`addr(s, mt) = lane_base + mt*(16*kK) ^ (s<<5)` for A and
`addr(s, nt) = lane_base + nt*(8*kK) ^ (s<<5)` for B. This replaced
runtime offset tables that spilled at 131 registers (~55 of 146 hot-loop
instructions were address math; cuBLAS's inner loop has ~0). Steady-state
read pointers advance one stage per iteration with an equality wrap,
replacing the per-k-tile `(tile % ring) * stage_bytes` recomputation
(UIMAD.WIDE magic-division ladder).

**Pipeline depth and barriers.** Every operand ring holds `kStages+1`
buffers: the load for tile `i+kStages` targets slot `(i-1)%(kStages+1)`,
which compute(i-1) finished reading before this iteration's barrier — no
post-compute barrier, one `__syncthreads` per k-tile. Prologue and tail
commits are unconditional so the group sequence stays tile-indexed and the
fixed `wait_group<kStages-1>` is iteration-invariant (a runtime
wait-count dispatch ladder cost 16 instructions/k-tile). A lean
`kStages`-deep ring trading the barrier for a 4th resident CTA measured
+5..9% slower at 1280³ and was removed.

**Crosswise loads.** Crosswise operands (A `[K][M]` / B `[N][K]` storage)
cannot cp.async into the canonical tile; they take the direct LDG.128×4 +
in-register PRMT transpose + STS.32 path. A staged variant (cp.async into
K-major staging + per-tile smem→smem transpose) measured 15-20% slower
across every probed shape including DRAM-streaming B (git history 5745c2f).

**Fast-loop peel.** When both operands are congruous, the whole CTA is
interior, base|ld is 16B-aligned and K has no tail, the mainloop switches
to a predication-free copy with loop-carried prefetch state: +4.5..10% on
the issue-bound 64×64 CTA (256³..1024³), −3% on the 128×128 CTA, so only
the small CTA opts in.

**Launch planning crossovers** (L20, TFLOPS, big vs alternative):
crosswise problems keep the 64×64 s3 CTA below ~1.5 waves of 128×128
tiles (M=256: 129.7 vs 113.1; 1024³: 107.2 vs 94.8; the big CTA wins from
M=640/1536³ on). Dual-congruous wave band picks narrow vs big by
`ceil(tiles/sm) * T_tile` with `T_narrow ≈ 0.53 * T_big` (M=384: 134.3 vs
114.4 narrow wins; M=1024: 202.5 vs 178.8 big wins). Sub-wave: narrow
wins past ~3/8 of a wave (1024³ 174 vs 131T), the big CTA's operand reuse
wins past ~5/8 (forcing 64×64 there cost 2048³ 123→171T). Non-128-divisible
shapes with 64-divisibility take the 64×64 CTA (edge tiles otherwise drag
the single wave; 1088³: 76 vs 93T). Persistent schedules (static
round-robin and atomic ticket) both measured worse on L20 (−4..−8%; the
ticket variant recovers L2 locality but its loop-head barrier costs what
the CTA-restart overlap saves).

**NN swap.** The dual-N-contiguous problem runs as its transpose
`E = B^T @ A^T` over swapped operands with an out-transposed epilogue
scatter (CUTLASS-sm90 `is_swapAB`): one instantiation fewer per tile
config, at the cost of a scalar-store scatter on a path no LLM-linear
operand pair hits.

## Build System

### Auto-detection

Kernels are built when **both** of these conditions are met:
1. `nvcc` is available on `PATH`
2. `torch.cuda.is_available()` returns `True`

Unless `CSRC_KERNELS=false` is set explicitly.

### Manual build

```bash
# During install
CSRC_KERNELS=true pip install -e . --no-build-isolation

# Rebuild after editing .cu/.cuh files
CSRC_KERNELS=true python setup.py build_ext --inplace
# Output: astrai/extension/lib/*.so

# Or invoke CMake directly
cmake -S csrc -B build/cmake \
  -DTORCH_HOME=<site-packages>/torch \
  -DPYTHON_INCLUDE_DIR=<python include> \
  -DPY_SOABI=cpython-312-x86_64-linux-gnu
cmake --build build/cmake -j 16
```

### Architecture flags

`setup.py` passes the GPU compute capability to CMake via `ASTRAI_CUDA_ARCH`. When
unset, `setup.py` auto-detects the real GPU capability through
`torch.cuda.get_device_capability()`; the CMake fallback default is `80` (sm_80):

- **sm_80+** (Ampere and later): enables the tensor-core MMA path
  (`mma.sync.m16n8k16.bf16` for bf16 attention, `mma.sync.m16n8k32` for FP8).
- **sm_89+**: required for the FP8 family (`fp8_ops`) — FP8 tensor-core
  instructions only exist on Ada/Hopper and newer. On older architectures,
  CMake emits a warning and skips the `fp8_ops` target so the remaining CUDA
  kernels still build successfully.
- **`-DASTRAI_NO_MMA`** is a manual escape hatch only — the build never defines
  it automatically. To disable the MMA path, add it to `NVCC_FLAGS` yourself;
  all supported build targets are sm_80+.

### Build configuration

`csrc/CMakeLists.txt` defines the CUDA extension build:

```
NVCC_FLAGS = -O3 --expt-relaxed-constexpr --use_fast_math
             --ptxas-options=-O3,-v --extra-device-vectorization --threads=16
```

Each kernel in `astrai/extension/lib` is compiled as an independent pybind11 module (one `.so` per kernel, named `<kernel>.cpython-*-x86_64-linux-gnu.so`). CMake builds all registered kernel targets in parallel via `cmake --build -j N` (the five base targets always; `fp8_ops` additionally on sm_89+). The target list is the **single source of truth**: `KERNEL_NAMES` and the parallel `KERNEL_SRCS` list in `csrc/CMakeLists.txt`; `astrai/extension/loader.py` auto-discovers the compiled `.so` files.

## Python Extension Architecture

The Python extension package separates low-level kernel bindings from execution
policy:

```text
astrai/extension/
├── __init__.py             # Stable public API
├── loader.py               # Optional compiled-module discovery and loading
├── ops/
│   ├── attention.py        # Stateless attention kernel wrappers
│   ├── rotary.py           # Stateless rotary kernel wrapper
│   ├── gemv.py             # Stateless BF16 GEMV primitive
│   ├── swiglu.py           # Stateless fused BF16 SwiGLU primitive
│   └── fp8.py              # Stateless FP8 primitives (custom_op)
├── fp8.py                  # FP8 strategy layer (fp8_autocast, recipes)
└── backend/
    ├── attention.py        # Backend selection, KV cache I/O, and fallback
    ├── swiglu.py           # Inference-only fused/unfused SwiGLU policy
    └── rotary.py           # Per-call CUDA/torch rotary dispatch
```

The dependency direction is one-way:

```text
model / inference
       |
       v
extension public API
       |
       v
backend policy  --->  ops wrappers  --->  loader  --->  compiled .so
       |
       +----------->  torch / flash-attn fallback
```

`ops` must not import `backend`. This keeps direct kernel bindings independent
of model, cache, fallback, and backend-selection policy.

### Ops Layer

`astrai.extension.ops` is the low-level boundary around compiled extensions:

- Wrappers are stateless and map Python arguments to pybind or
  `torch.library.custom_op` calls.
- Wrappers validate kernel availability and raise `RuntimeError` when a
  requested extension was not built.
- Wrappers do not choose another implementation, gather KV cache entries, or
  decide whether an input is supported by a backend.
- Tests that specifically exercise a compiled kernel may import from
  `astrai.extension.ops`.

For example, `attn_prefill(...)` means "run this CUDA kernel" rather than "run
attention using the best available implementation":

```python
from astrai.extension.ops import attn_prefill

output = attn_prefill(q, k, v, mask=mask, is_causal=True)
```

If the kernel is unavailable, this call fails. Callers that need fallback and
capability dispatch must use the public `attention(...)` entry point instead.

### Backend Layer

`astrai.extension.backend` owns execution policy:

- It selects CUDA, FlashAttention, or torch-native attention.
- It checks per-call constraints such as dtype, shape, head dimension, cache
  availability, and installed optional dependencies.
- It owns KV cache writes and reads because those operations differ by backend.
- It provides torch fallbacks and raises when an explicitly requested backend
  cannot handle a call.
- Rotary dispatch follows the same boundary without a backend class: the
  policy layer chooses the fused op for supported inference calls and otherwise
  uses the autograd-compatible torch implementation.

Normal model and inference code should import the stable API from
`astrai.extension`:

```python
from astrai.extension import ATTN_BACKEND, attention, attn_backend

output = attention(q, k, v, kv_cache=cache, layer_id=layer_id, fwd="decode")

with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
    output = attention(q, k, v)
```

The package root re-exports the supported high-level API and selected direct
kernel wrappers. Internal code should use `astrai.extension.backend` only when
it needs a backend type or policy implementation, and `astrai.extension.ops`
only when it deliberately requires one exact kernel.

### Placement Rules

When extending this package:

| Change | Location |
|--------|----------|
| Add a pybind call for a compiled kernel | `astrai/extension/ops/` |
| Add argument translation required by the compiled ABI | `astrai/extension/ops/` |
| Add capability checks or implementation selection | `astrai/extension/backend/` |
| Add a torch or third-party fallback | `astrai/extension/backend/` |
| Add attention KV cache behavior | `astrai/extension/backend/attention.py` |
| Expose a supported user-facing symbol | `astrai/extension/__init__.py` |

Imports belong at module scope. Optional dependencies such as `flash_attn` may
use a module-level guarded import. Type-only imports that would create a runtime
cycle belong under `TYPE_CHECKING`.

## Attention Backend

`astrai/extension/backend/attention.py` provides the backend abstraction:

- **`AttentionBackend`** (ABC): `fwd_decode` / `fwd_prefill` abstract methods, `forward` dispatches by q_len
- **`CudaBackend`**: CUDA kernel dispatch — decode via `attn_paged_decode` (page_size=1), prefill via `attn_paged_prefill` (ragged batch, `qo_indptr` + `kv_indptr`). Default on GPU.
- **`FlashAttnBackend`**: Optional flash-attn dispatch via `flash_attn_varlen_func` over gathered flat K/V.
- **`TorchNativeBackend`**: SDPA with indirect KV cache gather (always-available fallback)

Default priority: cuda > flash > torch. Set ``ASTR_BACKEND=cuda|torch_native|flash``
to override the default.

Select a backend via context manager (mirrors `torch.nn.attention.sdpa_kernel`):

```python
from astrai.extension import attn_backend, ATTN_BACKEND

with attn_backend(ATTN_BACKEND.CUDA):
    engine.generate("hello")
```

The `attention(...)` policy entry point falls back to `FlashAttnBackend` (when
flash-attn is installed and supports the call) or `TorchNativeBackend` when the
automatically selected CUDA backend cannot handle an input. Resolution
precedence is: explicit `attn_backend(...)` context > `ASTR_BACKEND` env >
default. An explicit `attn_backend(...)` selection is strict and raises instead
of silently switching implementations; the env override (and the implicit
default) fall back to the first compatible backend when incapable. Training
calls (`fwd=None`, no KV cache) resolve by capability: the CUDA cache kernels
cannot run without a cache, so they fall back to flash (mask-free/causal calls
only) and finally to torch SDPA.

### Rotary Backend

`astrai/extension/backend/rotary.py` provides `apply_rotary_emb(x, (cos, sin))` with auto-dispatch:

- **CUDA path**: calls `rotary_emb` kernel directly when available, input is bf16 on CUDA, and `torch.is_grad_enabled()` is `False` (inference)
- **Torch fallback**: complex multiply (`torch.view_as_complex` → `torch.complex` multiply → `torch.view_as_real`), used during training (supports autograd) or when kernel unavailable

No context-manager switching needed — the dispatch is automatic per call.

## Python Wrappers

`astrai/extension/ops/attention.py` provides Python wrappers for each compiled attention kernel. Each wrapper calls its CUDA kernel directly and raises `RuntimeError` if the `.so` is not available. Fallback to torch SDPA is handled by the attention backend, not the wrapper functions.

`astrai/extension/ops/rotary.py` provides the wrapper for the rotary embedding kernel. Fallback to torch complex multiply is handled by `backend/rotary.py`.

Interface (all functions):
```
is_causal: True = causal mask; False = non-causal
mask:      2D [batch, kv_len] or 3D [batch, q_len, kv_len] (bool, True=keep)
```

Layout convention: all q/k/v are `[batch, seq_len, n_heads, head_dim]` (blhd). Scale is always `1/sqrt(head_dim)`.

### Q Scheduling and KV Addressing

Prefill separates Q work scheduling from KV storage:

- `DenseQSchedule` maps a rectangular grid directly with
  `batch = blockIdx.z` and `q_tile = blockIdx.x`.
- `PackedQSchedule` consumes a compact work map for a packed
  `[total_q, q_heads, head_dim]` tensor.
- `ContigKV` and `PagedKV` only provide KV lengths and translate logical KV
  positions into physical addresses. They do not schedule Q blocks.

For ragged Q lengths `[70, 10, 130]` and 64 rows per Q tile, cache binding
builds:

```text
qo_indptr       = [0, 70, 80, 210]
q_tile_to_batch = [0, 0, 1, 2, 2, 2]
q_tile_to_index = [0, 1, 0, 0, 1, 2]
```

Paged prefill launches (MMA path, GQA head packing):

```text
grid.x = num_q_tiles * HB   # HB = min(G, WARPS): q heads packed per block
grid.y = kv_heads * ceil(G / HB)
grid.z = 1
```

The tensor-core prefill kernel packs `HB = min(G, WARPS)` query heads of one
kv-head group into a block, so K/V tiles stream once per block instead of once
per q head (~HB× less global K/V traffic). Warp `w` handles head slot `w / WPH`
and 16-row chunk `w % WPH`, where `WPH = WARPS / HB`; `G = q_heads / kv_heads`
and `G = 1` (MHA) degenerates to the historical one-head-per-block layout.
Each host Q tile (64 rows, `Q_TILE_ROWS`) splits into `HB` packed blocks along
`grid.x`. Each block resolves its request and request-local row range in O(1):

```cpp
host_tile = blockIdx.x / HB;
batch = q_tile_to_batch[host_tile];
row_base = q_tile_to_index[host_tile] * 64 + (blockIdx.x % HB) * (64 / HB);
```

The kernel then uses `qo_indptr[batch]` for the packed Q base and adjacent
`qo_indptr` / `kv_indptr` entries for that request's Q and KV lengths. This
avoids the previous per-block linear scan over the batch, shared-memory
broadcast, mapping barrier, and upper-bound grid with potentially invalid
blocks.

## Standalone Testing

Each `csrc/tests/*.cu` file has the `nvcc` compile command in its header comment. Example:

```bash
nvcc -I csrc/kernels -arch=sm_89 -O3 --use_fast_math \
     --ptxas-options=-O3,-v --extra-device-vectorization \
     -Xcompiler -fopenmp csrc/tests/attn_test.cu -o /tmp/test && /tmp/test
```

Test files:
- `attn_test.cu` — decode + prefill kernels (correctness tables + benchmarks)
- `attn_paged_test.cu` — paged decode/prefill kernels
- `fp8_test.cu` — single-warp bf16→fp8→mma.sync sanity check + full FP8 GEMM correctness (sm_89)

## Benchmarks

Hardware: NVIDIA L20 (sm_89, 46 GB), CUDA 12.8, driver 570.86.

Reproduce (decode + prefill in `attn_test.cu`, paged in `attn_paged_test.cu`):
```bash
nvcc -I csrc/kernels -arch=sm_89 -O3 --use_fast_math \
     --ptxas-options=-O3,-v --extra-device-vectorization \
     -Xcompiler -fopenmp csrc/tests/attn_test.cu -o /tmp/test && /tmp/test
```

## Known Optimization Targets

- **Decode D=256**: spill eliminated (BC=16 + STAGES=2), but still 248 regs — further tiling could help.
- **Prefill single-batch**: bandwidth low (22 GB/s at q=kv=2048) — compute-bound at ~94 TFLOP/s (near L20 bf16 ceiling ~193 TFLOP/s for non-causal).
- **Decode single-batch**: bandwidth low (113 GB/s at kv=512, 13% of 864 GB/s theoretical) — small kv underutilizes SMs despite split-KV; scales to 757 GB/s (88%) at B=16+.

## File Layout

```
csrc/
├── CMakeLists.txt                    # CMake build: kernel registry (KERNEL_NAMES / KERNEL_SRCS), torch/pybind11 linking
├── kernels/
│   ├── common/                       # cross-family pure-CUDA helpers (no torch)
│   │   ├── device.cuh                #   sm_at_least(), kMinSmForFp8* constants
│   │   ├── mma.cuh                   #   shared mma_sync<InT> + mma_shape<InT> (bf16 m16n8k16 / fp8 m16n8k32) + ldmatrix_x2/x4<T>
│   │   ├── cp_async.cuh              #   cp.async 16B primitives (predicated copy, commit/wait groups)
│   │   └── reduce.cuh                #   warp_reduce_max, atomic_max_float
│   ├── attention/                    # attention family (module names keep the attn_* prefix)
│   │   ├── common.h                  #   AttentionParams POD, TensorLayout enum (BHLD/BLHD)
│   │   ├── warp_utils.cuh            #   warp reduction helpers
│   │   ├── layout_policies.cuh       #   KV addressing policies: DenseQSchedule/PackedQSchedule, ContigKV/PagedKV
│   │   ├── mma_utils.cuh             #   ldmatrix/pack helpers + online-softmax (bf16 mma via common/mma.cuh)
│   │   ├── entry_utils.cuh           #   torch binding helpers: DISPATCH_HEAD_DIM, pack_*_params
│   │   ├── dispatchers.cuh           #   pure-CUDA launchers: dispatch_decode/prefill (+paged), split-K math
│   │   ├── decode_split_kv.cuh       #   decode kernel, scalar (split-KV)
│   │   ├── decode_split_kv_mma.cuh   #   decode kernel, MMA + split-K
│   │   ├── prefill_split_q.cuh       #   prefill kernel, scalar (split-Q)
│   │   ├── prefill_split_q_mma.cuh   #   prefill kernel, MMA (split-Q, GQA head packing, packed/ragged Q schedule)
│   │   ├── decode.cu                 #   → module attn_decode
│   │   ├── prefill.cu                #   → module attn_prefill
│   │   ├── paged_decode.cu           #   → module attn_paged_decode
│   │   └── paged_prefill.cu          #   → module attn_paged_prefill
│   ├── rotary_emb.cu                  # rotary embedding (kernel + binding in one file) → module rotary_emb
│   └── fp8/                          # FP8 family (module name fp8_ops)
│       ├── common.h                  #   FP8Format enum, Fp8GemmTraits, FP8Params / FP8QuantizeParams PODs, layout tags (no torch)
│       ├── quantize.cuh              #   quantize kernels: vectorized + 32×32-tile transpose (out_layout 0/1/2) (no torch)
│       ├── gemm.cuh                  #   GEMM umbrella: kernel orchestrator + host launch planning (no torch)
│       ├── gemm/                     #   GEMM device layers (humming/CUTLASS-style split)
│       │   ├── policy.cuh            #     smem budget / occupancy hint + Fp8GemmPolicy
│       │   ├── load.cuh              #     operand loaders (swizzle, congruous cp.async, crosswise direct)
│       │   ├── scheduler.cuh         #     grouped/plain raster mapping
│       │   ├── mainloop.cuh          #     stage rings + pipelined mma.sync mainloop
│       │   └── epilogue.cuh          #     fused bias + bf16 scatter + copy-out
│       └── ops.cu                    #   binding only: validation, param packing, launch dispatch, pybind
└── tests/
    ├── test_utils.cuh                # Shared test utilities (now_ms, f2bf, bf2f, randf)
    ├── attn_test.cu                  # Decode + prefill kernels
    ├── attn_paged_test.cu            # Paged decode/prefill kernels
    └── fp8_test.cu                   # MMA demo + GEMM correctness across layouts/K tiles/ragged shapes
```

Compiled `.so` files are placed in `astrai/extension/lib/`, separate from Python source files.

> Document Update Time: 2026-08-29
