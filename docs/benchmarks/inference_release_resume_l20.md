# Inference runtime release/resume on NVIDIA L20

Implementation commit: `8aa8e175d9e848dcb85250ca5106e978f012a6df`

Environment: NVIDIA L20, PyTorch 2.11.0+cu128, CUDA 12.8, BF16. The
`astrai-1b` preset used 24 layers, hidden size 1536, four KV heads, batch size
4, prompt length 128, and four greedy decode tokens. CUDA graphs were disabled
to isolate the scheduler-owned KV and workspace lifecycle. Each context bound
was measured across five complete release/resume/output-parity cycles.

| Max context | Runtime footprint | Reclaimed | Reclaimed % | Release median | Resume median | Greedy parity |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 200.97 MiB | 192.85 MiB | 95.96% | 83.60 ms | 1.34 ms | 5/5 |
| 8,192 | 777.14 MiB | 769.01 MiB | 98.95% | 96.22 ms | 5.04 ms | 5/5 |
| 32,768 | 3,081.81 MiB | 3,073.68 MiB | 99.74% | 90.79 ms | 5.10 ms | 5/5 |

`release()` includes scheduler stop, Python reference collection, and CUDA
allocator cache release. `resume()` reconstructs the cache and executor; the
reported resume latency excludes the first generation after reconstruction.
The model-only allocation remained resident throughout every cycle.

Reproduce one cell with:

```bash
python scripts/tools/benchmark_inference_lifecycle.py \
  --preset astrai-1b \
  --batch-size 4 \
  --max-seq-len 32768 \
  --prompt-len 128 \
  --max-tokens 4 \
  --trials 5 \
  --no-cuda-graph
```

Raw results:

- `inference_release_resume_l20_2048.json`
- `inference_release_resume_l20_8192.json`
- `inference_release_resume_l20_32768.json`
