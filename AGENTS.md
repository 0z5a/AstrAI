# AGENTS.md — AstrAI Development Guide

## Quick commands

```bash
# Lint + format (only files you touched)
ruff check --fix scripts/tools/train.py ...
ruff format scripts/tools/train.py ...

# Full-project format check (CI gate)
ruff format --check .

# Tests
python -m pytest tests/ -x -q          # quick, stop on first failure
python -m pytest tests/ -v             # CI mode

# Install dev deps
pip install .[dev]                     # includes pytest, ruff, httpx2
```

## CI gates

- **Lint CI**: only `ruff format --check .` + `ruff check . --select I` (import-order only). Full `ruff check` is **not** enforced in CI.
- **Test CI**: `python -m pytest tests/ -v`, Python 3.12, CPU-only (no GPU agents).
- `scripts/eval/` has **978 pre-existing ruff violations** — ignore them in your changes.

## Script conventions (post-refactor)

All `scripts/tools/*.py` use **click** (not argparse):
- Each script defines a `*_command` click command (e.g. `train_command`, `server_command`).
- Can be run standalone: `python scripts/tools/train.py --config pretrain.yaml`
- Uses `"""One-line docstring."""` style matching the rest of the project.

## Train config

- Supports `--config pretrain.yaml` + CLI flag override (CLI wins).
- Supports `--dry-run` to validate without training.
- YAML sections map to CLI flag names directly (e.g. `training.batch_per_device` → `--batch_per_device`).

## .gitignore: deny-by-default

Everything is ignored by default (`*`), then whitelisted by patterns:
- `!astrai/**/*.py`, `!scripts/**/*.py`, `!tests/**/*.py`, `!csrc/**/*.{py,cu,h,cuh}`
- `!pyproject.toml`, `!setup.py`, `!README.md`, `!.github/**`
- New Python files in `astrai/`, `scripts/`, `tests/` get picked up automatically.
- New files at root or in other dirs need an explicit `!` rule.

## Dependencies

- `pyyaml` is declared in `pyproject.toml` (used by train.py `--config`), though it also comes transitively via `huggingface-hub`.
- `httpx2` is a **dev-only** dependency (required by `starlette.testclient` used in inference tests).
- CUDA extension (`csrc/kernels/`) builds only when `nvcc` is available; `pip install .` falls back gracefully on CPU.

## Environment

- Python 3.12+
- PyTorch 2.11 with cu128 (`extra-index-url` in pyproject.toml)
- 8×L20D (48GB) training setup with NV18 interconnect

## Project layout (key directories)

| Dir | Purpose |
|-----|---------|
| `astrai/` | Core library (model, trainer, dataset, inference, config) |
| `scripts/tools/` | CLI scripts (train, server, generate, preprocess, benchmark) |
| `scripts/eval/` | Evaluation scripts (pre-existing lint debt, not actively changed) |
| `csrc/kernels/` | Custom CUDA kernels (attention decode/prefill) |
| `tests/` | pytest suites, no GPU required |
| `data/pretrain_unified/` | Pretraining shards (.bin + meta.json per chunk) |
| `checkpoint/` | Training checkpoints (gitignored at runtime) |
| `params/` | Model params/tokenizer (gitignored at runtime) |
| `assets/docs/` | Documentation and design docs |
