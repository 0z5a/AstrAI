# Fused SwiGLU benchmark

`csrc/bench/benchmark_swiglu.py` compares the directly callable fused BF16
SwiGLU primitive with both `F.linear` and the existing two-GEMV chain. It covers
the native AstrAI 1B MLP plus LLaMA 2 7B/13B, LLaMA 3 8B, and GPT-NeoX 20B
up/gate shapes at M=1/2/4/8 in eager and CUDA Graph modes.

```bash
CUDA_VISIBLE_DEVICES=0 python csrc/bench/benchmark_swiglu.py \
  --output results/swiglu.json \
  --markdown-output results/swiglu.md \
  --m-values 1,2,4,8 --mode both \
  --warmup 20 --iterations 100 --trials 10
```

Each trial uses A-B-C-C-B-A ordering to balance clock, cache, and temperature
drift. The generated JSON records every timing sample, p50/p90/p99, CUDA launch
count, maximum/mean absolute error, and cosine similarity.

## L20 findings

Hardware was one NVIDIA L20 (sm_89), PyTorch 2.11.0+cu128, CUDA 12.8. The
existing GPU5 inference service remained resident (15.4 GiB) but idle at the
sampling boundaries; no process or container was stopped.

For AstrAI 1B `(N,K)=(6912,1536)`, CUDA Graph medians were:

| M | torch (ms) | GEMV chain (ms) | fused (ms) | vs best unfused |
|---:|---:|---:|---:|---:|
| 1 | 0.02564 | 0.02298 | 0.01375 | +67.13% |
| 2 | 0.02484 | 0.02628 | 0.01416 | +75.40% |
| 4 | 0.02507 | 0.03839 | 0.01806 | +38.82% |
| 8 | 0.02563 | 0.07007 | 0.03339 | -23.24% |

The wide traditional shapes are weight-bandwidth dominated. CTA reuse keeps
the fused primitive within roughly -1.2% to +0.9% of the best unfused chain,
so none is eligible for automatic selection. This negative crossover is kept
in the raw evidence rather than hidden by a favorable subset.

The real 24-layer AstrAI checkpoint was then run through `InferenceEngine`,
including scheduler, sampling, and CUDA Graph. A-B-B-A medians were:

| Batch | unfused (ms/step) | forced fused (ms/step) | throughput gain |
|---:|---:|---:|---:|
| 1 | 4.125 | 3.925 | +5.10% |
| 2 | 4.245 | 4.055 | +4.69% |
| 4 | 4.475 | 4.305 | +3.95% |

## Dispatch decision

Direct correctness stayed close (`max_abs <= 2.4e-4`, cosine approximately
1.0), but deterministic greedy generations changed at M=1, M=2, and M=4.

For that reason no SM89 shape is enabled in `auto`. The default path stays on
the existing unfused linear backend, including any independently qualified
GEMV dispatch. `ASTRAI_SWIGLU=1` remains an explicit benchmark/experimentation
switch for callers that accept normal BF16 reduction-order variation. A future
automatic band must repeat both the performance and checkpoint-output gates.

## HBM re-measurement and kernel simplification

The operator numbers above are L2-resident: the AstrAI pair is 40.5 MB,
smaller than the 96 MB L2, so a tight timing loop re-reads warm weights
(13.75 us implies ~3.1 TB/s, far above the 864 GB/s spec). Real decode rotates
~1 GB of per-layer weights through L2 every step, so every call is cold.

Re-measuring with rotated weight copies (>= 240 MB working set) on the same
L20 showed:

- The fused CTA-reuse kernel sits at the dual-stream cold-read floor
  (702 vs 699 GB/s at (6912,1536); 369 vs 370 GB/s at (11008,4096)). Wide
  LLaMA matrices cap at ~370-400 GB/s regardless of kernel, even for a
  pure-read loop, so the old per-variant gaps there were noise.
- The `(6912,1536)` warp-per-row variant (formerly M=2/4/8) is 2-6% slower
  than CTA reuse at M=2/4 under cold weights and no longer wins at M=8 once
  the CTA drops to 128 threads. It and its dispatch table were deleted.
- New rule: 256 threads for M in [1, 7], 128 threads for M=8. End-to-end
  through the built module at (6912,1536): 738-752 GB/s for M in [1, 4] and
  702 GB/s at M=8 (+6% over the removed warp path).

The M=8 CUDA-Graph regression reported above (`-23.24%`) does not survive the
cold-weight regime: cuBLAS reaches L2 bandwidth in the warm loop while both
fused paths converge to the same HBM floor.
