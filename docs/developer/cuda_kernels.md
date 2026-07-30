# CUDA Kernels

AstrAI includes optional custom CUDA attention kernels for decode and prefill. These are built when `nvcc` is available and CUDA is detected, and are dispatched via the `CudaBackend` attention backend.

## Overview

| Kernel | File | Description |
|--------|------|-------------|
| `attn_decode` | `attn_decode.cu` | GQA decode attention (split-KV) |
| `attn_prefill` | `attn_prefill.cu` | GQA prefill attention (split-Q) |
| `attn_paged_decode` | `attn_paged_decode.cu` | Paged KV cache decode attention |

Additionally, optimized `.cuh` variants with tensor-core MMA (Matrix Multiply-Accumulate) exist:

| Variant | File | Optimization |
|---------|------|--------------|
| Split-KV MMA decode | `attn_decode_split_kv_mma.cuh` | Split KV across warps + MMA (sm_80+) |
| Split-Q MMA prefill | `attn_prefill_split_q_mma.cuh` | Split Q across warps + MMA (sm_80+) |
| Paged split-KV MMA decode | `attn_paged_decode_split_kv_mma.cuh` | Paged cache + split-KV + MMA |

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
# Output: astrai/extension/*.so
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

The `REGISTRY` in `csrc/build.py` lists all registered kernels (currently 3). Each entry maps a kernel name to its source files and build flags.

## Attention Backend

`astrai/extension/attention_backend.py` provides the backend abstraction:

- **`AttentionBackend`** (ABC): `fwd_decode` / `fwd_prefill` abstract methods, `forward` dispatches by q_len
- **`TorchNativeBackend`**: SDPA with indirect KV cache gather (default)
- **`CudaBackend`**: CUDA kernel dispatch — decode via `attn_paged_decode` (page_size=1), prefill via `attn_prefill`

Select a backend via context manager (mirrors `torch.nn.attention.sdpa_kernel`):

```python
from astrai.extension import attn_backend, ATTN_BACKEND

with attn_backend(ATTN_BACKEND.CUDA):
    engine.generate("hello")
```

`CudaBackend` falls back to `TorchNativeBackend` when a kernel is not available.

## Python Wrappers

`astrai/extension/attention_ops.py` provides Python wrappers for each compiled kernel. Each wrapper calls its CUDA kernel directly and raises `RuntimeError` if the `.so` is not available. Fallback to torch SDPA is handled by the attention backend, not the wrapper functions.

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
     csrc/tests/attn_decode_test.cu -o /tmp/test && /tmp/test
```

Test files:
- `attn_decode_test.cu` — basic decode kernel
- `attn_paged_decode_test.cu` — paged decode kernel
- `attn_prefill_test.cu` — prefill kernel

## Benchmarks

Hardware: NVIDIA L20 (sm_89, 46 GB), CUDA 12.8, driver 570.86.

Reproduce:
```bash
nvcc -I csrc -arch=sm_89 -O3 --use_fast_math \
     --ptxas-options=-O3,-v --extra-device-vectorization \
     csrc/tests/attn_<name>_test.cu -o /tmp/test && /tmp/test
```

## Known Optimization Targets

- **Decode D=256**: spill eliminated (BC=16 + STAGES=2), but still 248 regs — further tiling could help.
- **Prefill single-batch**: bandwidth low (52 GB/s at q=kv=2048) — likely compute-bound but near L20 bf16 ceiling (~94 TFLOP/s).
- **Decode single-batch**: bandwidth low (309 GB/s at kv=512) — L20 HBM ~864 GB/s theoretical; small kv underutilizes SMs despite split-KV.

## File Layout

```
csrc/
├── build.py              # Build system: REGISTRY, _arch_flags, nvcc flags
├── kernels/
│   ├── attn_common.h     # Shared attention utilities
│   ├── attn_decode.cu    # Basic decode kernel (registered)
│   ├── attn_prefill.cu   # Basic prefill kernel (registered)
│   ├── attn_paged_decode.cu             # Paged decode kernel (registered)
│   ├── attn_decode_split_kv.cuh         # Split-KV variant
│   ├── attn_decode_split_kv_mma.cuh     # Split-KV + MMA variant
│   ├── attn_prefill_split_q.cuh         # Split-Q variant
│   ├── attn_prefill_split_q_mma.cuh     # Split-Q + MMA variant
│   ├── attn_paged_decode_split_kv.cuh   # Paged + split-KV variant
│   ├── attn_paged_decode_split_kv_mma.cuh  # Paged + split-KV + MMA variant
│   ├── attn_dispatchers.cuh             # Kernel dispatch macros
│   ├── attn_entry_utils.cuh             # Entry point helpers
│   ├── attn_mma_utils.cuh               # MMA utilities
│   └── attn_warp_utils.cuh              # Warp-level utilities
└── tests/
    ├── test_utils.cuh           # Shared test utilities
    ├── attn_decode_test.cu      # Decode kernel test
    ├── attn_paged_decode_test.cu # Paged decode test
    └── attn_prefill_test.cu     # Prefill kernel test
```

> Document Update Time: 2026-07-30
