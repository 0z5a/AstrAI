# CUDA Kernels

AstrAI includes optional custom CUDA kernels for attention, rotary embedding,
and FP8 GEMM. These are built when `nvcc` is available and CUDA is detected.

## Overview

| Kernel | File | Description |
|--------|------|-------------|
| `attn_decode` | `attention/decode.cu` | GQA decode attention (split-KV) |
| `attn_prefill` | `attention/prefill.cu` | GQA prefill attention (split-Q) |
| `attn_paged_decode` | `attention/paged_decode.cu` | Paged KV cache decode attention |
| `attn_paged_prefill` | `attention/paged_prefill.cu` | Paged KV cache prefill attention (ragged batch) |
| `rotary_emb` | `rotary_emb.cu` | Fused rotary embedding (cos/sin lookup + rotation) |
| `quantize` | `quantize/quantize.cu` | FP8 quantization kernels (sm_89+) |
| `gemm` | `gemm/gemm.cu` | dtype-generic tensor-core GEMM binding (fp8 / W8A16 / W8A8 / W16A16) + the family's kernel-policy instantiation unit (sm_89+) |

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

The `quantize` family accelerates bf16 linear layers by
quantizing to FP8 and running tensor-core GEMMs (**requires sm_89+**; fp8
`mma.sync.m16n8k32` only exists on Ada/Hopper). Same three-layer style as
attention; the GEMM device code is split humming/CUTLASS-style into one
layered directory:

| File | Role |
|------|------|
| `quantize/common.h` | `FP8Format` enum (E4M3/E5M2) + `QuantLayout` + `QuantParams` POD — no torch |
| `quantize/quantize.cuh` | pure-CUDA device code: vectorized `fp8_quantize_kernel` + 32×32-tile transpose kernel (out_layout 0/1/2), `quant_in_traits<InT>` unpack — no torch |
| `quantize/dequant.cuh` | in-register dequantization functors (`DequantPair<SrcT, MmaT>`): the exact int8→bf16 expansion quantized-GEMM operands fold between the smem read and the mma |
| `gemm/common.h` | dtype-neutral GEMM family declarations: layout tags, `gemm_elem_traits<T>` (kBytes/kMmaK — adding a dtype = one specialization), `gemm_mma_traits<ElemA, ElemB>` (MmaT promotion + per-operand kDequantA/B), `GemmParams` POD |
| `gemm/policy.cuh` | dtype-generic `GemmTraits<ElemA, ElemB, CtaShape, WarpShape, Stages>` (tile geometry via the promoted MmaT) + `GemmTileConfig` (CUTLASS-style tile recipe: CTA/warp `Shape` types + stages + loop mode, with the named production manifest `TileBig128x128` / `TileBigFast` / `TileNarrow128x64` / `TileSmall64s2` / `TileSmall64s3`) + smem budget (`GemmSmem`) + `GemmPolicy` (dtypes × layouts × one tile config — the kernel's single template parameter); `Fp8GemmTraits`/`Fp8GemmPolicy` are fp8-format aliases |
| `gemm/load.cuh` | operand loaders: swizzle (`tile_at`), congruous cp.async (predicated + interior), `PrefetchCarry`, crosswise LDG+PRMT direct load, async trans staging |
| `gemm/scheduler.cuh` | CTA id → (block_m, block_n) grouped/plain raster (runtime `raster` knob) |
| `gemm/mainloop.cuh` | `GemmCollectiveMainloop`: stage rings, stage loads, fragment addressing (ldmatrix + dequantized scalar paths), pipelined mma.sync loop |
| `gemm/epilogue.cuh` | `GemmCollectiveEpilogue`: fused bias + per-row/per-channel scale folding + bf16 smem scatter + coalesced copy-out |
| `gemm/gemm.cuh` | umbrella: `gemm_kernel<Policy>` orchestrator + host planning (`plan_gemm` / `launch_plan`; 64×64 / 128×64 / 128×128 CTA) + entry `gemm_dispatch<ElemA, ElemB, OutT>` = `canonicalize_gemm` → `plan_gemm` → `launch_plan` |
| `quantize/quantize.cu` | binding only: `check_fp8_device` (sm_89+), param packing, launch dispatch, pybind → module `quantize` |

Scale semantics: `quantize` takes the quantization *multiplier*; the
strategy layer passes `scale.reciprocal()` and the kernel multiplies by it.
`mm_fp8` takes the combined dequant scale (`sa * sb`). `amax` is always
returned in the original input domain.

Python layer (two levels): `astrai/extension/ops/quantize.py` and
`ops/gemm.py` are the stateless kernel adapters (plain `quantize` /
`quantize_dual` / `mm_fp8` wrappers, one adapter file per compiled kernel
module), and `astrai/extension/quantize.py` is the strategy layer (fp8
recipes, delayed / dynamic scaling, `fp8_autocast`,
`fp8_linear_forward/backward` wiring `aten::linear` on CUDA, plus the int8
quantizers). The aten override installs lazily on the first fp8 activation
(autocast enter or the global enable), so importing the module is
dispatcher-neutral.

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
+5..9% slower at 1280³ and was removed. The final iteration carries no
trailing barrier of its own, so the epilogue transition pays one explicit
pair: `cp_async_wait_all()` (which drains only the calling thread's
groups) followed by `__syncthreads()` — without it a thread racing into
the epilogue scatters the output tile over peers' still-in-flight staging
writes and final fragment reads (caught by racecheck + a W8A8 stress
case: zeroed 32-row output bands on multi-wave grids).

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

**Tile vocabulary (CUTLASS-style).** Tile geometry is expressed as types,
not positional ints: `Shape<M, N, K>` (CTA tile; K = the per-stage k-tile)
and `Shape<M, N>` (warp tile) compose into a `GemmTileConfig` — one named
recipe bundling shapes + stage depth + loop mode. The production manifest
in `policy.cuh` (`TileBig128x128`, `TileBigFast`, `TileNarrow128x64`,
`TileSmall64s2/s3`) is the full set `launch_plan` dispatches to; a new
geometry in the ladder is one alias plus one planner branch, never a
re-spelled int list. Device collectives only read the derived
`Traits::kBlockM/kBlockN/...` constants, so this is purely a configuration
surface — the generated SASS is unchanged.

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

#### Quantized GEMM: W8A16 / W8A8 / W16A16 (humming-style)

The same mainloop serves every dtype pairing through one promotion rule
(`gemm_mma_traits`): the mma runs on the **MmaT** — symmetric fp8 keeps its
native `m16n8k32`, symmetric bf16 (W16A16) passes through untouched, and
any pair involving int8 (W8A16 weight-only, W8A8 dynamic, or the mirrored
A8W16) promotes to bf16 `m16n8k16` with per-operand in-register dequant
(`kDequantA` / `kDequantB` — W8A8 inserts both sides, W8A16 only B).
Staging never changes: int8 operands ride the existing congruous
cp.async / crosswise PRMT paths into the canonical swizzled tiles, and
`kMmaK` follows the promoted type so the tile geometry is shared.

**Dequant (quantize/dequant.cuh).** Each fragment register pair costs one
`LDS.16` + four LOP3-class instructions, exact for the full int8 range
including −128. bf16 carries only 7 mantissa bits, so the naive
"OR the byte into a bf16 base" trick (0x6400-style, exact for fp16) breaks
linearity — bit 7 spills into the exponent. Instead the magnitude bits
(0-6) and the sign bit (7) take separate LOP3s:
`h = (u & 0x7F) | 0x4300` → exactly 128+u7; `s = (u & 0x80) | 0x4300` →
128 or 256 as the sign picks; `v = h - s` is the exact int8 value, and
every intermediate is bf16-exact. A future humming-style offline byte
interleave (per k16 slice, u32 word c holding `(k₂c, k₂c₊₈, k₂c₊₁, k₂c₉)`)
would let one `LDS.32` feed both registers of a pair and drop the spread
PRMT; deferred until measurement justifies a repack pass.

**Scales.** `GemmParams` carries per-operand dequant scales folded
multiplicatively into the epilogue (the mma accumulates the raw quantized
product): per-tensor device scalar, per-row activation `a_scale[m]`, or
per-channel weight `b_scale[n]`. Grouped-along-K scales belong in the
mainloop and are not implemented. The transposed-output epilogue branch
applies `b_scale`/bias per kernel row — including the +8 accumulator half
(its own row factor), a pre-existing mixup the first scale-carrying
NN-swap test exposed.

**Python surface (two layers).** `astrai/extension/ops/gemm.py` is the
compiled `gemm` module's adapter — the stateless kernel entries (`mm_fp8`
and `mm_w8a16` / `mm_w8a8` / `mm_w16a16`, dtype pairing + scale extent
validated in the binding); `astrai/extension/quantize.py` carries the int8
policy (symmetric per-channel weight quantization, per-row dynamic
activation quantization). There is deliberately no nn.Module layer on the
int8 path: the only model-facing quantization integration is the fp8
autocast (same module, routing `aten::linear`), and quantized callers
compose the primitives directly.

Benchmark (L20, llama weight shapes, `csrc/bench/benchmark_w8.py`,
M=2048/4096): W8A16 reaches 0.83–1.08× of cuBLAS bf16 `F.linear`
(28–42 TFLOPS, faster than bf16 on the wide up_gate shapes where halved
weight traffic pays), W8A8 0.73–0.86×, W16A16 0.83–0.94× — the dequant
insert is not the bottleneck at these shapes; the modes track the
W16A16 baseline within a few percent.

**Humming parity (what we deliberately have and have not).** Adopted from
humming: in-register LOP3 dequant, the dtype-promotion unified mainloop,
per-operand epilogue scale placement, and now the CUTLASS-style
`Shape`/`GemmTileConfig` vocabulary. Not adopted, in rough priority order
for future work: grouped-along-K / 2-D block scales (GPTQ/AWQ import —
needs mainloop scale application, the epilogue cannot fold them),
asymmetric quantization with zero-points (offline folding at repack time),
sub-int8 dtypes (int4 and 3/5/6/7-bit need packed staging + a second
dequant family), the offline weight interleave (documented deferred above),
stream-K (wave-quantization; persistent scheduling alone measured worse
here), and the sm90+ feature set (TMA/cluster/warp-spec/PDL — a different
device target). Out of scope by design: NVRTC JIT + per-SM heuristic
tables (conflicts with the AOT single-instantiation-TU discipline) and MoE
gather/grouped GEMM. We keep two things humming lacks: strided-batch
operands with broadcast, and fp32 output.

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
- **sm_89+**: required for the FP8 family (`quantize`) — FP8 tensor-core
  instructions only exist on Ada/Hopper and newer. On older architectures,
  CMake emits a warning and skips the `quantize` target so the remaining CUDA
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

Each kernel in `astrai/extension/lib` is compiled as an independent pybind11 module (one `.so` per kernel, named `<kernel>.cpython-*-x86_64-linux-gnu.so`). CMake builds all registered kernel targets in parallel via `cmake --build -j N` (the five base targets always; `quantize` additionally on sm_89+). The target list is the **single source of truth**: `KERNEL_NAMES` and the parallel `KERNEL_SRCS` list in `csrc/CMakeLists.txt`; `astrai/extension/loader.py` auto-discovers the compiled `.so` files.

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
│   └── fp8.py              # Stateless FP8 primitives (custom_op)
├── fp8.py                  # FP8 strategy layer (fp8_autocast, recipes)
└── backend/
    ├── attention.py        # Backend selection, KV cache I/O, and fallback
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

- **`AttentionBackend`** (ABC): single abstract `forward`; each subclass branches on `fwd` ("decode" / "prefill" / None) internally, `_check_fwd` guards unknown modes
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
│   ├── quantize/                        # quantize family (no torch)
│   │   ├── common.h                  #   FP8Format enum, QuantLayout, QuantParams POD
│   │   ├── dequant.cuh               #   in-register dequant functors (DequantPair<SrcT, MmaT>: exact int8→bf16)
│   │   └── quantize.cuh              #   quantize kernels: vectorized + 32×32-tile transpose (out_layout 0/1/2)
│   ├── gemm/                         # GEMM family, dtype-neutral (→ module gemm)
│   │   ├── common.h                  #   layout tags, gemm_elem_traits<T>, gemm_mma_traits (MmaT promotion), GemmParams POD (no torch)
│   │   ├── gemm.cuh                  #   GEMM umbrella: kernel orchestrator + host launch planning (no torch)
│   │   ├── policy.cuh                #     Shape/TileConfig tile recipes + smem budget + GemmPolicy
│   │   ├── load.cuh                  #     operand loaders (swizzle, congruous cp.async, crosswise direct, trans staging)
│   │   ├── scheduler.cuh             #     grouped/plain raster mapping
│   │   ├── mainloop.cuh              #     stage rings + pipelined mma.sync mainloop (+ dequantized fragment paths)
│   │   ├── epilogue.cuh              #     fused bias + scale folding + bf16 scatter + copy-out
│   │   └── gemm.cu                   #   binding + explicit instantiations (mm_fp8 / mm_w8a16 / mm_w8a8 / mm_w16a16)
│   └── quantize/quantize.cu                  #   binding only (module quantize): validation, param packing, launch dispatch, pybind
└── tests/
    ├── test_utils.cuh                # Shared test utilities (now_ms, f2bf, bf2f, randf)
    ├── attn_test.cu                  # Decode + prefill kernels
    ├── attn_paged_test.cu            # Paged decode/prefill kernels
    └── fp8_test.cu                   # MMA demo + GEMM correctness: fp8/bf16/W8A16/W8A8/A8W16 across layouts/K tiles/ragged shapes
```

Compiled `.so` files are placed in `astrai/extension/lib/`, separate from Python source files.

> Document Update Time: 2026-08-29
