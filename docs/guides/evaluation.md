# Evaluation

AstrAI provides 7 evaluation scripts in `scripts/eval/` covering code generation, knowledge QA, perplexity, summarization, data quality, instruction following, and weight analysis.

## Overview

| Script | Metric | Model Invocation | External Dataset |
|--------|--------|-------------------|-------------------|
| `evaluate_humaneval.py` | Code-gen pass@1/10/100 | `InferenceEngine.generate` | HF `openai/openai_humaneval` (auto-download) |
| `evaluate_mmlu.py` | MCQ accuracy (log-likelihood) | Direct `model()` forward | HF `cais/mmlu` (auto-download) |
| `evaluate_ppl.py` | Perplexity / token loss | Direct `model()` forward | User JSONL |
| `evaluate_rouge.py` | ROUGE-1/2/L | None (pure metric) | User JSONL |
| `evaluate_ifd.py` | Instruction-Following Difficulty | Direct `model()` forward | User JSONL |
| `evaluate_ifeval.py` | Instruction-following constraints | `InferenceEngine.generate` | HF `google/IFEval` (auto-download) |
| `analyze_weights.py` | SVD effective rank / weight stats | None (loads safetensors) | Checkpoint dir |

Two invocation patterns exist:
- **Generation benchmarks** (HumanEval, IFEval): use `InferenceEngine` to generate responses, then score them.
- **Scoring benchmarks** (MMLU, PPL, IFD): call `model()` directly under `torch.inference_mode()` for log-likelihood computation.

Common defaults: `--param_path` defaults to `./params`; dtype defaults to `bfloat16` on CUDA, `float32` on CPU.

---

## HumanEval (Code Generation)

Generates completions for 164 programming problems, executes them against hidden tests, and reports pass@k.

```bash
python scripts/eval/evaluate_humaneval.py \
    --param_path ./params \
    --num_samples 20 \
    --batch_size 32 \
    --max_tokens 512 \
    --output results/humaneval.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | `./params` | Model directory |
| `--data_path` | `./humaneval/HumanEval.jsonl` | HumanEval JSONL (auto-downloaded if missing) |
| `--output` | None | Save results JSON (also writes `_completions.json`) |
| `--test_only` | None | Test an existing completions JSON (skip generation) |
| `--generate_only` | False | Only generate, skip execution/testing |
| `--num_samples` | 200 | Completions per problem (pass@k needs >= k) |
| `--max_tokens` | 512 | Max generation length |
| `--temperature` | 0.8 | Sampling temperature |
| `--top_p` | 0.95 | Nucleus sampling threshold |
| `--top_k` | 50 | Top-k sampling |
| `--batch_size` | 32 | Generation batch size |
| `--test_workers` | 8 | ProcessPoolExecutor workers for test execution |
| `--test_timeout` | 3.0 | Per-subprocess timeout (seconds) |
| `--problems` | None | Restrict to specific problem indices |

**Output**: stdout prints `pass@1`, `pass@10`, `pass@100`. With `--output`, writes per-problem results + `_summary` aggregate and a `_completions.json` file.

**Data**: Auto-downloads `openai/openai_humaneval` from HuggingFace on first run. Each problem has `task_id`, `entry_point`, `prompt`, `test`.

---

## MMLU (Knowledge QA)

57-subject multiple-choice accuracy via log-likelihood comparison. Supports n-shot few-shot prompting and option permutation.

```bash
python scripts/eval/evaluate_mmlu.py \
    --param_path ./params \
    --n_shot 5 \
    --subjects math_algebra history_us \
    --output results/mmlu.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | `./params` | Model directory |
| `--data_dir` | `./mmlu_data` | MMLU data directory (per-subject CSVs) |
| `--download` | False | Force re-download |
| `--n_shot` | 5 | Few-shot examples (0 = zero-shot) |
| `--subjects` | all 57 | Specific subjects to evaluate |
| `--output` | None | Output JSON path |
| `--split` | `test` | `test` or `val` |
| `--device` | auto | Device (`cuda` / `cpu`) |
| `--dtype` | auto | `bfloat16` on CUDA, `float32` on CPU |
| `--seed` | 0 | Seed for option permutation (0 = enabled, -1 = disabled) |

**How it works**: For each question, builds a prompt with n-shot examples, then scores each choice (A/B/C/D) by computing the summed log-likelihood of the choice token given the context. The choice with the highest log-prob is the prediction.

**Output**: stdout prints per-subject accuracy and overall. With `--output`, writes per-subject `{accuracy, correct, total}` + `_overall` aggregate.

**Data**: Auto-downloads `cais/mmlu` from HuggingFace. Stored as per-subject CSVs in `<data_dir>/<split>/` and `<data_dir>/dev/` (for few-shot).

---

## Perplexity (PPL)

Token-level negative-log-likelihood and perplexity on arbitrary text data. Supports streaming mode (memory-efficient) and non-streaming mode (exact per-token stats).

```bash
python scripts/eval/evaluate_ppl.py \
    --param_path ./params \
    --input_path data.jsonl \
    --output_dir ppl_results/ \
    --batch_size 4 \
    --max_length 2048
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | required | Model directory |
| `--input_path` | required | Input file, glob, or directory |
| `--output_dir` | required | Output directory for `summary.json` + token JSONL |
| `--text_key` | `text` | Key for the text field in input data |
| `--batch_size` | 4 | Batch size |
| `--max_length` | 2048 | Max sequence length (tokens) |
| `--token_level` | False | Store per-token log_probs + token-type analysis |
| `--max_samples` | None | Random subsample per file |
| `--device` | auto | Device |
| `--dtype` | auto | Torch dtype |

**Input**: JSONL or JSON files. Each item must have a field named by `--text_key` (default `text`). If `--input_path` is a directory, recursively collects `*.jsonl` and `*.json`.

**Output**: `summary.json` with per-file stats (tokens, mean/median loss, perplexity, p50/p90/p95/p99). With `--token_level`, also writes per-token JSONL with token IDs and log-probs.

---

## ROUGE

ROUGE-1/2/L (precision, recall, F1) for summarization. Self-contained implementation with no external dependencies.

```bash
python scripts/eval/evaluate_rouge.py \
    --data_path predictions.jsonl \
    --output results/rouge.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | required | JSONL with `reference`/`candidate` per line |
| `--output` | None | Output JSON path |

**Input**: JSONL, one object per line:
```json
{"reference": "Ground truth text", "candidate": "Model output text"}
```

**Output**: stdout prints `rouge-1`, `rouge-2`, `rouge-l` each as P/R/F1. With `--output`, writes JSON with `aggregate` and `per_item` scores.

Can also be imported as a library:
```python
from scripts.eval.evaluate_rouge import compute_rouge
scores = compute_rouge(reference, candidate)
```

---

## IFD (Instruction-Following Difficulty)

Data quality metric: `IFD = L_conditional / L_unconditional`. Measures how much harder it is to predict a response given its instruction vs. without it. Useful for filtering instruction-tuning data.

```bash
python scripts/eval/evaluate_ifd.py \
    --param_path ./params \
    --input_path sft_data.jsonl \
    --output_dir ifd_results/ \
    --format messages \
    --batch_size 8
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | required | Model directory |
| `--input_path` | required | Input file, glob, or directory |
| `--output_dir` | required | Output directory |
| `--max_len` | 2048 | Max token length |
| `--format` | `plain` | `plain` (instruction/response fields) or `messages` (chat format) |
| `--instr_key` | `instruction` | Instruction field key (plain format) |
| `--resp_key` | `response` | Response field key (plain format) |
| `--batch_size` | 8 | Items per model-forward flush |
| `--device` | auto | Device |
| `--dtype` | auto | Torch dtype |
| `--sentinel_text` | `\n` | Prefix for unconditional pass (`""` → bos/pad fallback) |
| `--per_token` | False | Include per-token IFD breakdown |
| `--max_samples` | None | Random subsample per file |

**How it works**: Two forward passes per batch — (1) conditional: packed BFD sequence with context + response, (2) unconditional: response prefixed with a sentinel. IFD = mean_conditional_loss / mean_unconditional_loss. IFD > 1 means the instruction makes the response harder to predict (higher quality data).

**Output**: Per-file `<label>_ifd.jsonl` with IFD scores per item. `summary.json` aggregates per-file stats.

---

## IFEval (Instruction Following)

Google's IFEval benchmark: generates responses and verifies 27 types of constraints (keywords, format, length, case, punctuation, etc.).

```bash
python scripts/eval/evaluate_ifeval.py \
    --param_path ./params \
    --num_samples 1 \
    --max_tokens 512 \
    --output results/ifeval.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | `./params` | Model directory |
| `--data_path` | `./ifeval/input_data.jsonl` | IFEval JSONL (auto-downloaded if missing) |
| `--output` | None | Output JSON path |
| `--max_tokens` | 512 | Max generation tokens |
| `--temperature` | 0.1 | Sampling temperature (low for instruction-following) |
| `--top_p` | 0.95 | Top-p sampling |
| `--top_k` | 50 | Top-k sampling |
| `--num_samples` | 1 | Samples per problem (best-of-n scoring) |
| `--batch_size` | 1 | Inference batch size |
| `--limit` | None | Limit to first N problems (quick testing) |
| `--dump_responses` | None | Path to dump raw responses as JSONL |

**Output**: stdout prints overall accuracy + per-constraint-type accuracy table. With `--output`, writes per-problem results + `_summary`.

**Data**: Auto-downloads `google/IFEval` from HuggingFace. Each problem has `key`, `prompt`, `instruction_id_list`, `kwargs`.

---

## Weight Analysis

SVD-based effective rank and weight statistics for checkpoint diagnostics. Does not load the model graph or run any forward pass.

```bash
python scripts/eval/analyze_weights.py \
    --ckpt_dir ./checkpoint/epoch_1_step_6000 \
    --output results/weights.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ckpt_dir` | required | Checkpoint dir with `model.safetensors` + `config.json` |
| `--compare` | None | Additional checkpoint dirs to compare |
| `--no_svd` | False | Skip SVD; show only weight stats (faster) |
| `--output` | None | Save results as JSON |
| `--device` | `cuda` | Device for SVD |

**Output**: SVD effective rank by component (ER@90/95/99%, entropic rank, condition number), per-layer effective rank grid, and weight value statistics (mean/std/min/max). Provides a utilization verdict (HIGH >0.85 / MODERATE >0.5 / LOW).

---

## Tips

- **Quick test**: Use `--limit` (IFEval) or `--problems` (HumanEval) to run on a small subset first.
- **Auto-download**: HumanEval, MMLU, and IFEval auto-download their datasets on first run. The other scripts expect user-provided data.
- **Output formats**: `--output` writes a single JSON for most scripts. PPL and IFD write an `--output_dir` containing `summary.json` plus per-file artifacts.
- **CPU mode**: All scripts auto-detect CUDA. To force CPU, use `--device cpu --dtype float32`.

> Document Update Time: 2026-07-30
