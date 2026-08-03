# CUDA Kernels

AstrAI includes optional custom CUDA kernels for attention and rotary embedding. These are built when `nvcc` is available and CUDA is detected, and are dispatched via the `CudaBackend` attention backend or auto-dispatched for rotary.

## Overview

| Kernel | File | Description |
|--------|------|-------------|
| `attn_decode` | `attn_decode.cu` | GQA decode attention (split-KV) |
| `attn_prefill` | `attn_prefill.cu` | GQA prefill attention (split-Q) |
| `attn_paged_decode` | `attn_paged_decode.cu` | Paged KV cache decode attention |
| `attn_paged_prefill` | `attn_paged_prefill.cu` | Paged KV cache prefill attention (ragged batch) |
| `rotary_emb` | `rotary_emb.cu` | Fused rotary embedding (cos/sin lookup + rotation) |

Additionally, optimized `.cuh` variants with tensor-core MMA (Matrix Multiply-Accumulate) exist:

| Variant | File | Optimization |
|---------|------|--------------|
| Split-KV MMA decode | `attn_decode_split_kv_mma.cuh` | Split KV across warps + MMA (sm_80+) |
| Split-Q MMA prefill | `attn_prefill_split_q_mma.cuh` | Split Q across warps + MMA (sm_80+) |
| Paged split-KV MMA decode | `attn_paged_decode_split_kv_mma.cuh` | Paged cache + split-KV + MMA |

### Rotary Embedding Kernel

The `rotary_emb` kernel (`csrc/kernels/rotary_emb.cu`) fuses cos/sin lookup and rotation into a single kernel:

- One thread per (head, dim-pair), vectorized `__nv_bfloat162` load/store
- f32 cos/sin input, bf16 compute and output
- 256-thread blocks, grid-stride loop
- Auto-dispatched via `apply_rotary_emb` in `astrai/extension/rotary_backend.py` (CUDA when available + inference mode, else torch complex-multiply fallback)
- No context-manager backend needed — rotary is backend-agnostic, both attention backends benefit

Standalone benchmark vs torch complex-multiply (48 calls = 24 layers × q+k): 6-9x faster, max diff 0 (decode) to 3e-2 (large prefill, bf16).

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
```

### Architecture flags

`csrc/build.py` auto-detects the GPU compute capability and generates the appropriate `nvcc` gencode flag:

- **sm_80+** (Ampere and later): enables tensor-core MMA path (`mma.sync.m16n8k16.bf16`)
- **Below sm_80**: adds `-DASTRAI_NO_MMA` to disable the MMA path at compile time

### Build configuration

```
NVCC_FLAGS = -O3 --expt-relaxed-constexpr --use_fast_math
             --ptxas-options=-O3,-v --extra-device-vectorization --threads=8
```

The `REGISTRY` in `csrc/build.py` lists all registered kernels (currently 5). Each entry maps a kernel name to its source files and build flags.

## Attention Backend

`astrai/extension/attention_backend.py` provides the backend abstraction:

- **`AttentionBackend`** (ABC): `fwd_decode` / `fwd_prefill` abstract methods, `forward` dispatches by q_len
- **`TorchNativeBackend`**: SDPA with indirect KV cache gather (default)
- **`CudaBackend`**: CUDA kernel dispatch — decode via `attn_paged_decode` (page_size=1), prefill via `attn_paged_prefill` (ragged batch, `qo_indptr` + `kv_indptr`)

Select a backend via context manager (mirrors `torch.nn.attention.sdpa_kernel`):

```python
from astrai.extension import attn_backend, ATTN_BACKEND

with attn_backend(ATTN_BACKEND.CUDA):
    engine.generate("hello")
```

`CudaBackend` falls back to `TorchNativeBackend` when a kernel is not available.

### Rotary Backend

`astrai/extension/rotary_backend.py` provides `apply_rotary_emb(x, (cos, sin))` with auto-dispatch:

- **CUDA path**: calls `rotary_emb` kernel directly when available, input is bf16 on CUDA, and `torch.is_grad_enabled()` is `False` (inference)
- **Torch fallback**: complex multiply (`torch.view_as_complex` → `torch.complex` multiply → `torch.view_as_real`), used during training (supports autograd) or when kernel unavailable

No context-manager switching needed — the dispatch is automatic per call.

## Python Wrappers

`astrai/extension/attention_ops.py` provides Python wrappers for each compiled attention kernel. Each wrapper calls its CUDA kernel directly and raises `RuntimeError` if the `.so` is not available. Fallback to torch SDPA is handled by the attention backend, not the wrapper functions.

`astrai/extension/rotary_ops.py` provides the wrapper for the rotary embedding kernel. Fallback to torch complex multiply is handled by `rotary_backend.py`.

Interface (all functions):
```
is_causal: True = causal mask; False = non-causal
mask:      2D [batch, kv_len] or 3D [batch, q_len, kv_len] (bool, True=keep)
```

Layout convention: all q/k/v are `[batch, seq_len, n_heads, head_dim]` (blhd). Scale is always `1/sqrt(head_dim)`.

## Standalone Testing

Each `csrc/tests/*.cu` file has the `nvcc` compile command in its header comment. Example:

```bash
nvcc -I csrc -arch=sm_89 -O3 --use_fast_math \
     --ptxas-options=-O3,-v --extra-device-vectorization \
     -Xcompiler -fopenmp csrc/tests/attn_test.cu -o /tmp/test && /tmp/test
```

Test files:
- `attn_test.cu` — decode + prefill kernels (correctness tables + benchmarks)
- `attn_paged_test.cu` — paged decode/prefill kernels

## Benchmarks

Hardware: NVIDIA L20 (sm_89, 46 GB), CUDA 12.8, driver 570.86.

Reproduce (decode + prefill in `attn_test.cu`, paged in `attn_paged_test.cu`):
```bash
nvcc -I csrc -arch=sm_89 -O3 --use_fast_math \
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
├── build.py              # Build system: REGISTRY, _arch_flags, nvcc flags
├── kernels/
│   ├── attn_common.h     # Shared attention params (AttentionParams, PagedAttentionParams)
│   ├── attn_decode.cu    # Basic decode kernel (registered)
│   ├── attn_prefill.cu   # Basic prefill kernel (registered)
│   ├── attn_paged_decode.cu             # Paged decode kernel (registered)
│   ├── attn_paged_prefill.cu            # Paged prefill kernel (registered)
│   ├── rotary_emb.cu                    # Fused rotary embedding kernel (registered)
│   ├── attn_decode_split_kv.cuh         # Split-KV variant
│   ├── attn_decode_split_kv_mma.cuh     # Split-KV + MMA variant
│   ├── attn_prefill_split_q.cuh         # Split-Q variant
│   ├── attn_prefill_split_q_mma.cuh     # Split-Q + MMA variant
│   ├── attn_paged_decode_split_kv.cuh   # Paged + split-KV variant
│   ├── attn_paged_decode_split_kv_mma.cuh  # Paged + split-KV + MMA variant
│   ├── attn_paged_prefill_split_q.cuh   # Paged + split-Q variant
│   ├── attn_paged_prefill_split_q_mma.cuh  # Paged + split-Q + MMA variant
│   ├── attn_dispatchers.cuh             # Kernel dispatch macros
│   ├── attn_entry_utils.cuh             # Entry point helpers
│   ├── attn_mma_utils.cuh               # MMA utilities
│   └── attn_warp_utils.cuh              # Warp-level utilities
└── tests/
    ├── test_utils.cuh           # Shared test utilities
    ├── attn_test.cu             # Decode + prefill kernels
    └── attn_paged_test.cu       # Paged decode/prefill kernels
```

Compiled `.so` files are placed in `astrai/extension/lib/`, separate from Python source files.

> Document Update Time: 2026-07-31
