# Containerized Training Deployment

AstrAI uses one training YAML as the declaration for both host-side container
runtime settings and in-container training settings. Do not invoke the trainer
with raw `docker compose up`; use `scripts/train.sh` so preflight validation,
checkpoint recovery, and graceful shutdown remain active.

## Architecture

```text
train.yaml
  ├── runtime                parsed on the host before Docker starts
  └── model/data/...         parsed by train.py inside the container
        │
scripts/train.sh             preflight, Compose wrapper, lifecycle, timer
  └── docker-compose.yml     GPU passthrough, mounts, image, container limits
        └── scripts/docker/train-entrypoint.sh   process count, parallel mode, auto-resume
              └── train.py --config /run/astrai/train.yaml
```

The two parsers deliberately own different sections. `scripts/docker/train_runtime.py`
reads only `runtime`; `scripts/tools/train.py` reads only
`model/data/parallel/training/ckpt/log`. Explicit trainer arguments after `--`
override training YAML values.

## Runtime Schema

```yaml
runtime:
  job_name: astrai-train
  paths:
    data: ./data
    model: ./params
    checkpoints: ./checkpoints
  gpu:
    devices: all
    parallel_mode: auto  # one GPU: none; multiple GPUs: ddp
  container:
    cuda_tag: cu128
    ipc: host
    stop_grace_period: 10m
    stop_timeout_seconds: 600
    checkpoint_keep_last: 5
    # max_duration_hours: 12
  # Add host-specific workarounds only when required:
  # environment:
  #   NCCL_P2P_DISABLE: "1"
  #   NCCL_NET_GDR_LEVEL: "0"
```

- Relative paths resolve from the YAML file's directory, not the current shell.
- `devices` is either `all` or a non-empty physical GPU index list. Compose
  passes all GPUs once; `CUDA_VISIBLE_DEVICES` performs the only filtering.
- The process count is derived from `devices`. With `all`, the entrypoint uses
  `torch.cuda.device_count()` after Docker starts.
- `parallel_mode: auto` selects `none` for one GPU and `ddp` for multiple GPUs.
  Use `fsdp` explicitly when model sharding is required.
- To select specific physical GPUs, replace `all` with a list such as
  `devices: [0, 1]`.
- `environment` values are explicitly passed to the training container. Keep
  host-specific NCCL workarounds here; they are not universal defaults.
- `max_duration_hours` starts a detached host timer that calls the same graceful
  `stop` command. A manual stop cancels the timer.

## Fixed Container Paths

| Runtime path | Container path | Access |
|---|---|---|
| `runtime.paths.data` | `/data` | read-only |
| `runtime.paths.model` | `/models/base` | read-only |
| `runtime.paths.checkpoints` | `/checkpoints` | read-write |
| the selected YAML | `/run/astrai/train.yaml` | read-only |

Training configuration must therefore use `data_root_path: /data`. The source
code is baked into `/app`; `start` reuses the existing image, so run
`bash scripts/train.sh build [CONFIG]` after code changes.

## Operations

The config argument defaults to `./train.yaml`:

```bash
bash scripts/train.sh init [CONFIG]
bash scripts/train.sh preflight [CONFIG]
bash scripts/train.sh start [CONFIG]
bash scripts/train.sh start [CONFIG] --foreground -- --dry-run
bash scripts/train.sh logs [CONFIG]
bash scripts/train.sh status [CONFIG]
bash scripts/train.sh stop [CONFIG]
bash scripts/train.sh restart [CONFIG]
bash scripts/train.sh clean [CONFIG] --keep 5
bash scripts/train.sh clean [CONFIG] --keep 5 --force
```

`init` creates the declared runtime directories but does not generate or mutate
the YAML. `preflight` validates Docker, paths, base model files, checkpoint
writability, GPU configuration, and rendered Compose configuration.

## Checkpoint Recovery

Checkpoints are stored below
`runtime.paths.checkpoints/<job_name>/epoch_<N>_step_<N>`. A checkpoint is
complete only when it contains:

```text
meta.json
config.json
model.safetensors
optimizer.pt
scheduler.pt
```

`start` resumes the latest complete checkpoint and ignores partial writes. If no
complete checkpoint exists, `/models/base/config.json` and
`/models/base/model.safetensors` are required. `stop` sends `SIGTERM`; the
trainer finishes at a batch boundary and saves an emergency checkpoint before
the Docker timeout expires.

## Hard Rules

1. Keep Docker settings in `runtime` and trainer settings in the remaining YAML sections.
2. Filter GPUs once: Compose passes `count: all`; `devices` becomes `CUDA_VISIBLE_DEVICES`.
3. Do not force DDP for a model that requires FSDP; declare the mode explicitly.
4. Do not use `kill -9` for routine shutdown; use `scripts/train.sh stop CONFIG`.
5. The image user is built with the host UID/GID so mounted checkpoints retain usable ownership.

> Document Update Time: 2026-08-22
