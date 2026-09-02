# Decode linear shape benchmark

`scripts/tools/benchmark_gemv.py` records the `F.linear` baseline used to decide
whether a BF16 GEMV or small-M kernel should enter automatic inference dispatch.
It does not change model execution or select a custom kernel.

The default matrix covers the AstrAI 1B q/k/v/out projections, MLP up/gate/down,
and LM head for `M=1,2,4,8,16,32`. Each shape runs in eager and CUDA Graph replay
modes. Results include device-event latency samples, p50/p90/p99, estimated
effective IO bandwidth, and CUDA kernel launches per call.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tools/benchmark_gemv.py \
  --output results/decode_linear.json \
  --markdown-output results/decode_linear.md
```

Use `--shape NAME:N:K` repeatedly to override the preset and `--m-values` to
change the decode batch sizes. Compare each GPU architecture only with its own
baseline; do not use absolute A100-versus-L20 numbers as a dispatch criterion.
Keep the raw JSON as the source of truth and generate tables with
`--markdown-output` rather than transcribing measurements by hand.

For direct A/B coverage of the custom kernel and guarded dispatcher across
traditional LLaMA and GPT-NeoX decode shapes, use:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/tools/benchmark_gemv_common.py \
  --suite all --family traditional --m 2 4 \
  --output results/gemv_common.json
```

The kernel suite compares the directly callable primitive with `F.linear`.
Use repeatable `--shape-label` and `--chain-label` filters for a focused run.
The synthetic-chain suite alternates `ASTRAI_GEMV=0` and `auto`, includes
dependent MLP work and Python dispatch, and rotates through distinct weights.
Automatic dispatch is keyed on the decode batch size alone (`M` in `[2, 4]` on
compute capability 8.0+); use `--candidate-mode 1` to characterize a family
before widening that band. The checked-in final evidence always uses `auto`.
It is deliberately not labeled a whole-model throughput benchmark. Both
suites report median/p90 CUDA-event latency plus maximum absolute error,
relative L2 error, and row-wise argmax parity.
