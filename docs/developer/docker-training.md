# Containerized Training Deployment

Rules for running AstrAI distributed training in containers, distilled from real deployment failures. Read before touching `Dockerfile`, `docker-compose.yml`, `scripts/train.sh`, `train-entrypoint.sh`. AGENTS.md mirrors this locally; this file is the committed version.

## Architecture

```
scripts/train.sh           host-side CLI: env loading, preflight, compose wrapper, lifecycle
  └── docker-compose.yml   GPU passthrough, mounts, in-container env vars, entrypoint
        └── train-entrypoint.sh   GPU-count resolution, parallel-mode selection, auto-resume
              └── train.py --config /run/astrai/train.yaml
```

| Layer | Responsible for | NOT responsible for |
|-------|-----------------|---------------------|
| `train.sh` | host paths, `.env.train`, preflight, lifecycle | training args, GPU selection, parallel mode |
| compose | GPU passthrough, mounts, in-container env (NCCL) | training args (beyond `TRAIN_*` forwarding) |
| entrypoint | `--ckpt_dir/--nprocs/--parallel_mode/--param_path`, resume | hyperparameters (YAML/CLI) |
| `train.yaml` | hyperparameters (`_merge_yaml_into_kwargs`, CLI wins) | container paths, process count |

## Path Conventions

| Host var | Container | Perm | Purpose |
|---|---|---|---|
| `TRAIN_DATA_DIR` | `/data` | ro | dataset (`data_root_path` must be `/data`) |
| `TRAIN_MODEL_DIR` | `/models/base` | ro | base model (`config.json` + `model.safetensors`) |
| `TRAIN_CHECKPOINT_DIR` | `/checkpoints` | rw | checkpoint root, per-`TRAIN_JOB_NAME` subdirs |
| `TRAIN_CONFIG_FILE` | `/run/astrai/train.yaml` | ro | training YAML (mounted only on `start`) |
| code | `/app` | image | **not a mount**; rebuild image for code changes |

## Hard Rules

1. **Filter GPUs once**: compose passes the full physical set (`count: all`); `CUDA_VISIBLE_DEVICES` filters inside by physical index. Never `count: N` + physical indices (double filter leaves 1 card → `device_id out of range`).
2. **In-container UID = host UID**: Dockerfile builds the user via `USER_UID/USER_GID` args; `train.sh` injects `ASTRAI_UID/GID` (bash `UID` is readonly). compose `user:` alone does not create the /etc/passwd entry — torch's `getpass.getuser()` then dies with `uid not found`.
3. **In-container env vars are explicit**: `.env.train` (`--env-file`) is only compose's interpolation dictionary — never reaches the container. A var arrives only via a value-less `environment` entry (`- VAR`, read from the calling process env).
4. **NCCL hang workaround** (this host): `NCCL_P2P_DISABLE=1` + `NCCL_NET_GDR_LEVEL=0` must be in-container.
5. **Checkpoint complete =** `meta.json + config.json + model.safetensors + optimizer.pt + scheduler.pt`; `start` auto-resumes the latest complete one.
6. **tqdm is silent without a TTY**: add `disable=False` in `astrai/trainer/train_callback.py`; `metric.jsonl` (per step) works as progress evidence regardless.

## Operations

```bash
bash scripts/train.sh init        # first run: dirs + .env.train (edit per machine)
bash scripts/train.sh preflight   # validate Docker/paths/GPU/model/YAML/compose
bash scripts/train.sh start       # build + start in background (auto-resume)
bash scripts/train.sh start --foreground -- --dry-run   # print plan only
bash scripts/train.sh logs | status | stop | restart
bash scripts/train.sh clean --keep 5   # prune old checkpoints (--force to delete)
```

## Files

- `docker-compose.yml` — trainer service: `count: all`, `ASTRAI_UID/GID` build args + `user:`, env whitelist, mounts
- `Dockerfile` — production stage builds user from `USER_UID/USER_GID`; `ENV HOME=/home/astrai`; `USER astrai`
- `scripts/train.sh` — `load_env` filters `UID=` lines (readonly var); `compose()` injects `ASTRAI_UID/GID`
- `scripts/docker/train-entrypoint.sh` — GPU-count resolution, parallel mode, resume
- `.env.train`, `train.yaml` — host-specific; templates from `scripts/train.sh init`; scientific-notation floats (`2e-5`) parse correctly since train.py uses the YAML 1.2 float schema
