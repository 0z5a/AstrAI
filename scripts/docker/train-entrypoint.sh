#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/train-common.sh"

TRAIN_JOB_NAME="${TRAIN_JOB_NAME:?TRAIN_JOB_NAME is required}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${TRAIN_JOB_NAME}"
BASE_MODEL="${BASE_MODEL:-/models/base}"
TRAIN_CONFIG="${TRAIN_CONFIG:-}"
TRAIN_GPU_COUNT="${TRAIN_GPU_COUNT:-all}"

validate_job_name "${TRAIN_JOB_NAME}"
if [[ "${TRAIN_GPU_COUNT}" == "all" ]]; then
    TRAIN_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
[[ "${TRAIN_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] || die "No visible GPU found"
if [[ -n "${TRAIN_CONFIG}" ]]; then
    [[ -f "${TRAIN_CONFIG}" ]] || die "Training config not found: ${TRAIN_CONFIG}"
fi
[[ -r /data ]] || die "Training data directory is not readable: /data"

mkdir -p "${CHECKPOINT_DIR}"
[[ -w "${CHECKPOINT_DIR}" ]] || die "Checkpoint directory is not writable: ${CHECKPOINT_DIR}"

latest_checkpoint="$(find_latest_checkpoint "${CHECKPOINT_DIR}" || true)"

train_args=(
    python scripts/tools/train.py
    --ckpt_dir "${CHECKPOINT_DIR}"
    --nprocs "${TRAIN_GPU_COUNT}"
)

if [[ -n "${TRAIN_CONFIG}" ]]; then
    train_args+=(--config "${TRAIN_CONFIG}")
fi

if (( TRAIN_GPU_COUNT > 1 )); then
    train_args+=(--parallel_mode ddp)
else
    train_args+=(--parallel_mode none)
fi

if [[ -n "${latest_checkpoint}" ]]; then
    log_info "Resuming ${TRAIN_JOB_NAME} from ${latest_checkpoint}"
    train_args+=(--param_path "${latest_checkpoint}" --resume)
else
    [[ -s "${BASE_MODEL}/config.json" ]] || die "Base model config not found: ${BASE_MODEL}/config.json"
    [[ -s "${BASE_MODEL}/model.safetensors" ]] || die "Base model weights not found: ${BASE_MODEL}/model.safetensors"
    log_info "Starting ${TRAIN_JOB_NAME} from ${BASE_MODEL}"
    train_args+=(--param_path "${BASE_MODEL}")
fi

log_info "GPUs=${TRAIN_GPU_COUNT}, checkpoints=${CHECKPOINT_DIR}"

# Replace the shell so the container init forwards SIGTERM to the trainer.
exec "${train_args[@]}" "$@"
