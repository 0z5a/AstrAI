# CUDA Operators

AstrAI includes optional custom CUDA kernels for attention, rotary
embedding, and the quantized GEMM family. They are built when `nvcc` is
available and CUDA is detected. This folder is the home of the
per-operator documentation — math contract first, then design notes; the
family-wide infrastructure (build system, python extension layers,
testing, file layout) lives at the bottom of this file.

## Overview
| Kernel | File | Description |
|--------|------|-------------|
| `attn_decode` | `attention/decode.cu` | GQA decode attention (split-KV) |
| `attn_prefill` | `attention/prefill.cu` | GQA prefill attention (split-Q) |
| `attn_paged_decode` | `attention/paged_decode.cu` | Paged KV cache decode attention |
| `attn_paged_prefill` | `attention/paged_prefill.cu` | Paged KV cache prefill attention (ragged batch) |
| `rotary_emb` | `rotary_emb.cu` | Fused rotary embedding (cos/sin lookup + rotation) |
| `quantize` | `quantize/quantize.cu` | FP8 quantization kernels (sm_89+) |
| `gemm` | `gemm/gemm.cu` + per-dtype-pair `gemm_*.cu` | dtype-generic tensor-core GEMM binding + one explicit `gemm_dispatch` instantiation per dtype pair (fp8 / W8A16 / W8A8 / W16A16, sm_89+) |

Additionally, optimized `.cuh` variants with tensor-core MMA (Matrix Multiply-Accumulate) exist:

| Variant | File | Optimization |
|---------|------|--------------|
| Split-KV MMA decode | `attention/decode_split_kv_mma.cuh` | Split KV across warps + MMA (sm_80+) |
| Split-Q MMA prefill | `attention/prefill_split_q_mma.cuh` | Split Q across warps + MMA (sm_80+) |

## Operator index

| Operator | Doc | Kernel module | Python entry |
|---|---|---|---|
| Quantize (FP8) | [quantize.md](quantize.md) | `csrc/kernels/quantize/` | `astrai/extension/ops/quantize.py`; strategy layer `astrai/extension/quantize.py` (`fp8_autocast`, aten::linear override) |
| GEMM / Linear (bf16 · fp8 · w8a16 · w8a8) | [gemm.md](gemm.md) | `csrc/kernels/gemm/` | adapter `astrai/extension/ops/gemm.py` |
| Attention (decode / paged / split-Q prefill, MMA variants) | [attention.md](attention.md) | `csrc/kernels/attention/` | `astrai/extension/ops/attention.py`; dispatch `astrai/extension/backend/attention.py` |
| Rotary embedding | [rotary.md](rotary.md) | `csrc/kernels/rotary_emb.cu` | `astrai/extension/ops/rotary.py`; dispatch `astrai/extension/backend/rotary.py` |

Conventions: every operator doc leads with its **math contract** — if the
contract and the kernel disagree, one of them is a bug, fixed in the same
commit as the kernel change. Performance numbers carry the measurement
discipline (production dispatch path, interleaved A/B rounds, median).
Required-reading gates: tile-vocabulary / planner / fp8-recipe changes →
[gemm.md](gemm.md); backend dispatch changes → the operator doc first.

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

`setup.py` passes the GPU compute capability to CMake via `ASTRAI_CUDA_ARCH`
(a semicolon list, e.g. `"80;89;120a"`, produces one multi-arch fatbin: one
image per token, the driver picks the slice per device). A token is `<CC
digits>` with an optional arch-specific `a` suffix; every numeric gate
compares the suffix-stripped value.

There is **no CMake default**: an unset or empty `ASTRAI_CUDA_ARCH` builds no
kernel targets at all, so a GPU-less configure installs the pure-Python
package and `loader.py` simply finds no `.so` files. When the env is unset,
`setup.py` auto-detects the real GPU through `torch.cuda.get_device_capability()`
(CC 12.0 reports `120a`, so a dev build gets the full-rate fp8 cell):

- **sm_80+** (Ampere and later): enables the tensor-core MMA path
  (`mma.sync.m16n8k16.bf16` for bf16 attention, `mma.sync.m16n8k32` for FP8).
- **sm_89+**: required for the FP8 family (`quantize`) — FP8 tensor-core
  instructions only exist on Ada/Hopper and newer. On older architectures,
  CMake emits a warning and skips the `quantize` target so the remaining CUDA
  kernels still build successfully.
- **sm_120a** (consumer Blackwell): the arch-specific pass that activates the
  fp8 `block_scale` cell. Listing plain `120` for the gemm target warns at
  configure time — the plain fp8 instruction decodes at half rate on sm_120,
  and the runtime route keys on the device's `cc == 120`, so the mismatch is
  silent without the warning.
- **`-DASTRAI_NO_MMA`** is a manual escape hatch only — the build never defines
  it automatically. To disable the MMA path, add it to `NVCC_FLAGS` yourself;
  all supported build targets are sm_80+.

### Build configuration

`csrc/CMakeLists.txt` defines the CUDA extension build:

```
NVCC_FLAGS = -O3 --expt-relaxed-constexpr --use_fast_math
             --ptxas-options=-O3,-v --extra-device-vectorization
             -Xfatbin --compress-all --threads=16
```

`--compress-all` stores every fatbin entry (SASS and PTX) compressed; the
driver decompresses at module load, so the executed code is unchanged and the
`.so` is ~4x smaller (measured per gemm TU: 8.03 MB -> 1.89 MB).

Each kernel in `astrai/extension/lib` is compiled as an independent pybind11 module (one `.so` per kernel, named `<kernel>.cpython-*-x86_64-linux-gnu.so`). CMake builds all registered kernel targets in parallel via `cmake --build -j N` (the five base targets always; `quantize` additionally on sm_89+; nothing at all when `ASTRAI_CUDA_ARCH` is unset). The target list is the **single source of truth**: `KERNEL_NAMES` and the parallel `KERNEL_SRCS` list in `csrc/CMakeLists.txt`; `astrai/extension/loader.py` auto-discovers the compiled `.so` files.

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
│   ├── fp8.py              # Stateless FP8 primitives (custom_op)
│   └── gemm.py             # Stateless quantized-GEMM wrapper + autotune hook
├── gemm_autotune.py        # Runtime plan-row autotuner (see below)
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
- `quant_gemm_test.cu` — quantized GEMM correctness: every dtype pair (fp8/int8/bf16 × layouts/K tiles/ragged shapes/scales/fp32 out) + a per-combo TFLOPS bench (sm_89+)

### Behavior-preservation gate (SASS digest)

A refactor that claims to change nothing must prove it: `csrc/bench/sass_digest.py`
hashes every kernel symbol's SASS (`cuobjdump -sass`) out of the compile-tree
`.o` files, with REG usage riding along, and `--compare` exits nonzero on any
added/removed/changed function. Device-side dead-code deletion and host-side
dedup gate on 658/658 identity instead of re-benchmarking (2026-09-19 audit).
The one drift class a POD layout change causes (param-field offsets shift,
ptxas re-selects load widths and renumbers registers) is adjudicated by
instruction-count + mnemonic-histogram equality and the bitwise pytest suite.
The same audit closed the sass digest for shared-helper extraction in hot
device code: pulling verbatim-duplicated straight-line blocks into
`__forceinline__` helpers moves ptxas scheduling across inline boundaries
(42 mixed-pair kernels drifted SASS). That family lands under a benchmark
gate instead (2026-09-20, `d521f14`): big-shape A-B-A-B x3 (ms-scale
kernels, ±0.3% noise) plus a 6-round alternating re-measurement of the
worst cell, with the sass-identical kernels as a noise control — measured
flat (1.00x on every combo, several mixed pairs slightly faster, matching
the -0.56% instruction delta and -1..-2 registers).

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
│   │   ├── device.cuh                #   DeviceFacts geometry query (sms / smem opt-in / L2); fp8 capability helpers live in quantize/common.h, the torch-bound gate in quantize/checks.h
│   │   ├── mma.cuh                   #   shared mma_sync<InT> + mma_shape<InT> (bf16 m16n8k16 / fp8 m16n8k32) + ldmatrix_x2<T> and the per-lane ldmatrix cores + typed fragment cells (AFrag/BFrag/CFrag, by-reference fma/ldmatrix overloads)
│   │   ├── pipeline.cuh              #   async data-movement vocabulary, one header: raw cp.async 16B emitters (fixed + runtime-src-size zfill) and the mbarrier PTX sites, plus the PipelineSync (sm_80/89 wait_group+syncthreads) stage pipeline
│   │   ├── swizzle.cuh               #   staging-layout vocabulary: Swizzle/Shape/Stride/Layout + composition(Swizzle, Layout) in 16B-chunk units; per-tile SmemLayout types declared by the gemm collectives
│   │   ├── tensor.cuh                #   tensor vocabulary, cute's Tensor<Engine, Layout>: PtrEngine/ArrayEngine, RingLayout/CellLayout, one Tensor type spelled directly (make_ring constructs the stage ring, stage_of slices a slot)
│   │   └── reduce.cuh                #   warp_reduce_max, atomic_max_float
│   ├── attention/                    # attention family (module names keep the attn_* prefix)
│   │   ├── common.h                  #   AttentionParams POD, TensorLayout enum (BHLD/BLHD)
│   │   ├── warp_utils.cuh            #   warp reduction helpers
│   │   ├── layout_policies.cuh       #   KV addressing policies: DenseQSchedule/PackedQSchedule, ContigKV/PagedKV
│   │   ├── mma_utils.cuh             #   ldmatrix/pack helpers + online-softmax (bf16 mma via common/mma.cuh)
│   │   ├── entry_utils.cuh           #   torch binding helpers: DISPATCH_HEAD_DIM, pack_*_params
│   │   ├── dispatchers.cuh           #   pure-CUDA launchers: dispatch_decode/prefill(_impl) funnel (+paged), split-K math
│   │   ├── decode_split_kv.cuh       #   decode kernel, scalar (split-KV)
│   │   ├── decode_split_kv_mma.cuh   #   decode kernel, MMA + split-K
│   │   ├── prefill_split_q.cuh       #   prefill kernel, scalar (split-Q)
│   │   ├── prefill_split_q_mma.cuh   #   prefill kernel, MMA (split-Q, GQA head packing, packed/ragged Q schedule)
│   │   ├── decode.cu                 #   → module attn_decode
│   │   ├── prefill.cu                #   → module attn_prefill
│   │   ├── paged_decode.cu           #   → module attn_paged_decode
│   │   └── paged_prefill.cu          #   → module attn_paged_prefill
│   ├── rotary_emb.cu                  # rotary embedding (kernel + binding in one file) → module rotary_emb
│   ├── quantize/                        # quantize family (pure CUDA; checks.h is the torch-bound gate)
│   │   ├── common.h                  #   sm_at_least + kMinSmForFp8 capability helpers, QuantLayout, QuantParams POD (raw __nv_fp8_* types)
│   │   ├── checks.h                  #   torch-bound entry validation (check_fp8_device over ATen-cached properties)
│   │   ├── dequant.cuh               #   in-register dequant functors (DequantPair<SrcT, MmaT>: exact int8→bf16)
│   │   └── quantize.cuh              #   quantize kernels: vectorized + 64×32-tile transpose (out_layout 0/1/2, Dual as a template param)
│   ├── gemm/                         # GEMM family, dtype-neutral (→ module gemm)
│   │   ├── common.h                  #   layout tags, gemm_elem_traits<T>, gemm_mma_traits (MmaT promotion), GemmParams POD (no torch)
│   │   ├── gemm.cuh                  #   GEMM umbrella: kernel orchestrator + host launch planning (no torch)
│   │   ├── policy.cuh                #     Shape/TileConfig tile recipes + smem budget + GemmPolicy + TileManifest (+ kTileClassCta)
│   │   ├── plan_table.h              #     AOT dispatch rows (TableRow): override/per-class builtin/degraded row sources
│   │   ├── load.cuh                  #     operand loaders (typed staged tiles over declared layouts, congruous cp.async + zfill, crosswise direct, trans staging, PrefetchCarry)
│   │   ├── scheduler.cuh             #     grouped/plain raster mapping
│   │   ├── mainloop.cuh              #     stage rings + pipelined mma.sync mainloop (+ dequantized fragment paths)
│   │   ├── epilogue.cuh              #     fused bias + scale folding + bf16/fp32 smem scatter + copy-out
│   │   ├── gemm_bf16_* / gemm_*.cu   #     per-pair explicit gemm_dispatch instantiation units (one nvcc job each)
│   │   └── gemm.cu                   #   quant_gemm binding + dtype-pair switch (module gemm; extern template decls)
│   └── quantize/quantize.cu                  #   binding only (module quantize): validation, param packing, launch dispatch, pybind
└── tests/
    ├── test_utils.cuh                # Shared test utilities (now_ms, f2bf, bf2f, randf)
    ├── attn_test.cu                  # Decode + prefill kernels
    ├── attn_paged_test.cu            # Paged decode/prefill kernels
    └── quant_gemm_test.cu           # GEMM correctness: fp8/bf16/int8 pairs across layouts/K tiles/ragged shapes + dtype-combo TFLOPS bench
```

Compiled `.so` files are placed in `astrai/extension/lib/`, separate from Python source files.

> Document Update Time: 2026-09-20
