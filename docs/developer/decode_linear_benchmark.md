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
