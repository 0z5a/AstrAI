#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/docker/lib/train-common.sh"

ENV_FILE="${TRAIN_ENV_FILE:-${ROOT_DIR}/.env.train}"
COMPOSE_BASE=(
    docker compose
    --project-directory "${ROOT_DIR}"
    --file "${ROOT_DIR}/docker-compose.yml"
    --profile train
)

usage() {
    cat <<'EOF'
Usage: scripts/train.sh <command> [options]

Commands:
  init                 Create local directories and .env.train
  preflight            Validate Docker, paths, GPU settings, and Compose
  build                Build the trainer image
  start [--foreground] [-- ARGS...] Start or resume training
  stop                 Gracefully stop and checkpoint training
  restart              Stop, then start training
  logs                 Follow trainer logs
  status               Show container and latest checkpoint status
  latest               Print the latest complete checkpoint path
  list                 List all complete checkpoints
  clean [--keep N]     Preview old checkpoint removal
  clean --force        Remove old checkpoints after previewing

Environment:
  TRAIN_ENV_FILE       Env file path (default: .env.train)
  TRAIN_CONFIG_FILE    Optional host YAML mounted only when the job starts

Training arguments come from an externally mounted TRAIN_CONFIG or ARGS passed
after --. The image does not contain experiment configuration.
EOF
}

load_env() {
    if [[ -f "${ENV_FILE}" ]]; then
        set -a
        # UID/GID are readonly in bash; compose gets them via ASTRAI_UID/GID in compose()
        # shellcheck disable=SC1090
        source <(grep -v -E '^[[:space:]]*(UID|GID)=' "${ENV_FILE}")
        set +a
    fi

    TRAIN_JOB_NAME="${TRAIN_JOB_NAME:-astrai-train}"
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-./data}"
    TRAIN_MODEL_DIR="${TRAIN_MODEL_DIR:-./params}"
    TRAIN_CHECKPOINT_DIR="${TRAIN_CHECKPOINT_DIR:-./checkpoints}"
    TRAIN_GPU_COUNT="${TRAIN_GPU_COUNT:-all}"
    TRAIN_STOP_TIMEOUT="${TRAIN_STOP_TIMEOUT:-600}"

    validate_job_name "${TRAIN_JOB_NAME}"
}

resolve_path() {
    if [[ "$1" = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "${ROOT_DIR}" "${1#./}"
    fi
}

# Read the optional top-level `infra:` section from TRAIN_CONFIG_FILE and
# export the host-side variables it overrides (job name, mount paths, GPU
# filter). Compose interpolation prefers the shell environment over the
# --env-file, so these exports win over .env.train; keys absent from `infra`
# fall back to the env file. Requires python3 with PyYAML on the host.
load_infra() {
    local infra_file exports

    [[ -n "${TRAIN_CONFIG_FILE:-}" ]] || return 0
    infra_file="$(resolve_path "${TRAIN_CONFIG_FILE}")"
    [[ -f "${infra_file}" ]] || return 0

    if ! command -v python3 >/dev/null 2>&1; then
        die "TRAIN_CONFIG_FILE is set but python3 is missing; it is needed to read the 'infra' section"
    fi
    if ! python3 -c 'import yaml' >/dev/null 2>&1; then
        die "TRAIN_CONFIG_FILE is set but PyYAML is missing on the host (install python3-yaml)"
    fi

    exports="$(TRAIN_INFRA_FILE="${infra_file}" python3 - <<'PYEOF'
import os
import shlex
import sys
import yaml

path = os.environ["TRAIN_INFRA_FILE"]
try:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
except Exception as exc:
    print(f"failed to parse {path}: {exc}", file=sys.stderr)
    sys.exit(1)

infra = cfg.get("infra") or {}
if not isinstance(infra, dict):
    print(f"the 'infra' section in {path} must be a mapping", file=sys.stderr)
    sys.exit(1)

mapping = {
    "job_name": "TRAIN_JOB_NAME",
    "data_dir": "TRAIN_DATA_DIR",
    "model_dir": "TRAIN_MODEL_DIR",
    "checkpoint_dir": "TRAIN_CHECKPOINT_DIR",
    "gpu_count": "TRAIN_GPU_COUNT",
    "cuda_visible_devices": "CUDA_VISIBLE_DEVICES",
}
for key, env_name in mapping.items():
    if key in infra:
        print(f"export {env_name}={shlex.quote(str(infra[key]))}")
PYEOF
)"
    if [[ -n "${exports}" ]]; then
        eval "${exports}"
        log_info "Applied infra overrides from ${infra_file}"
    fi

    validate_job_name "${TRAIN_JOB_NAME}"
}

checkpoint_dir() {
    printf '%s/%s\n' "$(resolve_path "${TRAIN_CHECKPOINT_DIR}")" "${TRAIN_JOB_NAME}"
}

compose() {
    local -a command=("${COMPOSE_BASE[@]}")

    if [[ -f "${ENV_FILE}" ]]; then
        command+=(--env-file "${ENV_FILE}")
    fi

    # Inject the host user into compose so container processes share the
    # checkpoint directory ownership (bash UID/GID are readonly).
    ASTRAI_UID="$(id -u)" ASTRAI_GID="$(id -g)" "${command[@]}" "$@"
}

init_environment() {
    local data_dir model_dir checkpoints_dir

    data_dir="$(resolve_path "${TRAIN_DATA_DIR}")"
    model_dir="$(resolve_path "${TRAIN_MODEL_DIR}")"
    checkpoints_dir="$(resolve_path "${TRAIN_CHECKPOINT_DIR}")"
    mkdir -p "${data_dir}" "${model_dir}" "${checkpoints_dir}"

    if [[ ! -f "${ENV_FILE}" ]]; then
        cat >"${ENV_FILE}" <<'EOF'
TRAIN_JOB_NAME=astrai-train
TRAIN_DATA_DIR=./data
TRAIN_MODEL_DIR=./params
TRAIN_CHECKPOINT_DIR=./checkpoints
TRAIN_CONFIG_FILE=
TRAIN_GPU_COUNT=all
# CUDA_VISIBLE_DEVICES=0,1
# TRAIN_* vars above can be overridden per-job via the top-level `infra:`
# section in TRAIN_CONFIG_FILE (see docs/developer/docker-training.md).
CUDA_TAG=cu128
TRAIN_IPC_MODE=host
TRAIN_STOP_GRACE_PERIOD=10m
TRAIN_STOP_TIMEOUT=600
CHECKPOINT_KEEP_LAST=5
EOF
        log_info "Created ${ENV_FILE}"
    else
        log_info "Keeping existing ${ENV_FILE}"
    fi
    log_info "Data: ${data_dir}"
    log_info "Model: ${model_dir}"
    log_info "Checkpoints: ${checkpoints_dir}"
}

preflight() {
    local data_dir model_dir checkpoints_dir config_file latest visible_count

    require_command docker
    docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
    [[ "${TRAIN_GPU_COUNT}" == "all" || "${TRAIN_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] ||
        die "TRAIN_GPU_COUNT must be 'all' or a positive integer"

    data_dir="$(resolve_path "${TRAIN_DATA_DIR}")"
    model_dir="$(resolve_path "${TRAIN_MODEL_DIR}")"
    checkpoints_dir="$(resolve_path "${TRAIN_CHECKPOINT_DIR}")"
    [[ -d "${data_dir}" ]] || die "Training data directory not found: ${data_dir}"
    mkdir -p "${checkpoints_dir}/${TRAIN_JOB_NAME}"
    [[ -w "${checkpoints_dir}/${TRAIN_JOB_NAME}" ]] || die "Checkpoint directory is not writable"

    if [[ -n "${TRAIN_CONFIG_FILE:-}" ]]; then
        config_file="$(resolve_path "${TRAIN_CONFIG_FILE}")"
        [[ -f "${config_file}" ]] || die "Training config not found: ${config_file}"
    fi

    latest="$(find_latest_checkpoint "${checkpoints_dir}/${TRAIN_JOB_NAME}" || true)"
    if [[ -z "${latest}" ]]; then
        [[ -s "${model_dir}/config.json" ]] || die "Model config not found: ${model_dir}/config.json"
        [[ -s "${model_dir}/model.safetensors" ]] || die "Model weights not found: ${model_dir}/model.safetensors"
    else
        log_info "Resume candidate: ${latest}"
    fi

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${TRAIN_GPU_COUNT}" != "all" ]]; then
        IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
        visible_count="${#visible_gpus[@]}"
        (( visible_count == TRAIN_GPU_COUNT )) ||
            die "TRAIN_GPU_COUNT=${TRAIN_GPU_COUNT}, but CUDA_VISIBLE_DEVICES exposes ${visible_count} GPU(s)"
    fi

    compose config --quiet
    log_info "Preflight passed for ${TRAIN_JOB_NAME} (GPU request: ${TRAIN_GPU_COUNT})"
}

start_training() {
    local foreground="$1"
    local config_file container running
    local -a run_options=()
    shift

    preflight
    if [[ -n "${TRAIN_CONFIG_FILE:-}" ]]; then
        config_file="$(resolve_path "${TRAIN_CONFIG_FILE}")"
        run_options+=(
            --volume "${config_file}:/run/astrai/train.yaml:ro"
            --env TRAIN_CONFIG=/run/astrai/train.yaml
        )
    elif [[ -z "${TRAIN_CONFIG:-}" && $# -eq 0 ]]; then
        die "Set TRAIN_CONFIG_FILE or pass complete trainer arguments after --"
    fi

    container="astrai-trainer-${TRAIN_JOB_NAME}"
    running="$(docker inspect --format '{{.State.Running}}' "${container}" 2>/dev/null || true)"
    [[ "${running}" != "true" ]] || die "Trainer is already running: ${container}"
    docker rm "${container}" >/dev/null 2>&1 || true
    if [[ "${foreground}" == "true" ]]; then
        compose run --build --rm "${run_options[@]}" trainer "$@"
    else
        compose run -d --build --name "${container}" \
            "${run_options[@]}" trainer "$@"
        log_info "Training started; run scripts/train.sh logs to follow it"
    fi
}

stop_training() {
    log_info "Stopping trainer with ${TRAIN_STOP_TIMEOUT}s grace period"
    docker stop --timeout "${TRAIN_STOP_TIMEOUT}" "astrai-trainer-${TRAIN_JOB_NAME}" >/dev/null 2>&1 ||
        log_warn "Trainer container is not running"
}

restart_training() {
    local container="astrai-trainer-${TRAIN_JOB_NAME}"

    docker inspect "${container}" >/dev/null 2>&1 ||
        die "Trainer container not found; use start with a config or CLI arguments first"
    log_info "Restarting trainer with ${TRAIN_STOP_TIMEOUT}s grace period"
    docker restart --timeout "${TRAIN_STOP_TIMEOUT}" "${container}" >/dev/null
}

show_status() {
    local latest

    docker ps -a --filter "name=^/astrai-trainer-${TRAIN_JOB_NAME}$"
    latest="$(find_latest_checkpoint "$(checkpoint_dir)" || true)"
    if [[ -n "${latest}" ]]; then
        log_info "Latest checkpoint: ${latest}"
    else
        log_info "No complete checkpoint found for ${TRAIN_JOB_NAME}"
    fi
}

clean_checkpoints() {
    local keep="$1" force="$2" dir count remove_count index path
    local -a checkpoints=()

    [[ "${keep}" =~ ^[1-9][0-9]*$ ]] || die "--keep must be a positive integer"
    dir="$(checkpoint_dir)"
    while IFS= read -r line; do
        [[ -n "${line}" ]] && checkpoints+=("${line#* * }")
    done < <(list_complete_checkpoints "${dir}")

    count="${#checkpoints[@]}"
    remove_count=$((count - keep))
    if (( remove_count <= 0 )); then
        log_info "Nothing to clean; ${count} complete checkpoint(s), keeping ${keep}"
        return
    fi

    for ((index = 0; index < remove_count; index++)); do
        path="${checkpoints[index]}"
        if [[ "${force}" == "true" ]]; then
            rm -rf -- "${path}"
            log_info "Removed ${path}"
        else
            printf 'Would remove %s\n' "${path}"
        fi
    done
    [[ "${force}" == "true" ]] || log_warn "Preview only; add --force to delete"
}

main() {
    local command="${1:-}" foreground=false keep="${CHECKPOINT_KEEP_LAST:-5}" force=false
    local -a train_args=()
    [[ -n "${command}" ]] || { usage; exit 1; }
    shift || true
    load_env
    load_infra

    case "${command}" in
        init)
            init_environment
            ;;
        preflight)
            preflight
            ;;
        build)
            preflight
            compose build trainer
            ;;
        start)
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --foreground)
                        foreground=true
                        shift
                        ;;
                    --)
                        shift
                        train_args=("$@")
                        break
                        ;;
                    *)
                        die "Unknown start option: $1 (put trainer arguments after --)"
                        ;;
                esac
            done
            start_training "${foreground}" "${train_args[@]}"
            ;;
        stop)
            stop_training
            ;;
        restart)
            restart_training
            ;;
        logs)
            docker logs -f --tail "${TRAIN_LOG_TAIL:-200}" "astrai-trainer-${TRAIN_JOB_NAME}"
            ;;
        status)
            show_status
            ;;
        latest)
            find_latest_checkpoint "$(checkpoint_dir)" || die "No complete checkpoint found"
            ;;
        list)
            list_complete_checkpoints "$(checkpoint_dir)" | while read -r _epoch _step path; do
                printf '%s\n' "${path}"
            done
            ;;
        clean)
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --keep)
                        [[ $# -ge 2 ]] || die "--keep requires a value"
                        keep="$2"
                        shift 2
                        ;;
                    --force)
                        force=true
                        shift
                        ;;
                    *)
                        die "Unknown clean option: $1"
                        ;;
                esac
            done
            clean_checkpoints "${keep}" "${force}"
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            die "Unknown command: ${command}"
            ;;
    esac
}

main "$@"
