# Fused SwiGLU benchmark

`scripts/tools/benchmark_swiglu.py` compares the directly callable fused BF16
SwiGLU primitive with both `F.linear` and the existing two-GEMV chain. It covers
the native AstrAI 1B MLP plus LLaMA 2 7B/13B, LLaMA 3 8B, and GPT-NeoX 20B
up/gate shapes at M=1/2/4/8 in eager and CUDA Graph modes.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tools/benchmark_swiglu.py \
  --output results/swiglu.json \
  --markdown-output results/swiglu.md \
  --m-values 1,2,4,8 --mode both \
  --warmup 20 --iterations 100 --trials 10
```

Each trial uses A-B-C-C-B-A ordering to balance clock, cache, and temperature
drift. The generated JSON records every timing sample, p50/p90/p99, CUDA launch
count, maximum/mean absolute error, and cosine similarity. The checked-in L20
raw run is `docs/benchmarks/swiglu_l20_sm89.json`.

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
including scheduler, sampling, and CUDA Graph. A-B-B-A medians from the raw
log in `docs/benchmarks/swiglu_engine_l20_sm89.txt` were:

| Batch | unfused (ms/step) | forced fused (ms/step) | throughput gain |
|---:|---:|---:|---:|
| 1 | 4.125 | 3.925 | +5.10% |
| 2 | 4.245 | 4.055 | +4.69% |
| 4 | 4.475 | 4.305 | +3.95% |

## Dispatch decision

Direct correctness stayed close (`max_abs <= 2.4e-4`, cosine approximately
1.0), but deterministic greedy generations changed at M=1, M=2, and M=4. The
M=1/2 hash pairs are preserved in
`docs/benchmarks/swiglu_greedy_m1_m2_l20_sm89.txt`, and the M=4 pair is in
`docs/benchmarks/swiglu_greedy_m4_l20_sm89.txt`.

For that reason no SM89 shape is enabled in `auto`. The default path stays on
the existing unfused linear backend, including any independently qualified
GEMV dispatch. `ASTRAI_SWIGLU=1` remains an explicit benchmark/experimentation
switch for callers that accept normal BF16 reduction-order variation. A future
automatic band must repeat both the performance and checkpoint-output gates.
