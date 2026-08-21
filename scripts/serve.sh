#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/docker/lib/train-common.sh"

COMPOSE_BASE=(
    docker compose
    --project-directory "${ROOT_DIR}"
    --file "${ROOT_DIR}/docker-compose.yml"
)

usage() {
    cat <<'EOF'
Usage: scripts/serve.sh <command> [CONFIG] [options]

CONFIG defaults to ./serve.yaml. The same file declares host runtime settings
under `runtime:` and server settings under `server:`.

Commands:
  init [CONFIG]                     Create the model directory
  preflight [CONFIG]                Validate Docker, paths, GPU, and Compose
  build [CONFIG]                    Build the serving image
  up [CONFIG]                       Start the server container (detached)
  run [CONFIG]                      Start the server container (foreground)
  down [CONFIG]                     Stop and remove the server container
  restart [CONFIG]                  Down, then up
  logs [CONFIG]                     Follow server logs
  status [CONFIG]                   Show container status
EOF
}

resolve_path() {
    if [[ "$1" = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "${ROOT_DIR}" "${1#./}"
    fi
}

load_config() {
    CONFIG_FILE="$(resolve_path "$1")"
    [[ -f "${CONFIG_FILE}" ]] || die "Serving config not found: ${CONFIG_FILE}"
    require_command python3
    python3 -c 'import yaml' >/dev/null 2>&1 ||
        die "PyYAML is required on the host (install python3-yaml)"

    local exports
    exports="$(python3 "${ROOT_DIR}/scripts/tools/serve_runtime.py" exports "${CONFIG_FILE}")" ||
        die "Failed to load runtime configuration"
    eval "${exports}"
    if [[ -n "${SERVE_JOB_NAME}" ]]; then
        validate_job_name "${SERVE_JOB_NAME}"
    fi
}

compose() {
    ASTRAI_UID="$(id -u)" ASTRAI_GID="$(id -g)" "${COMPOSE_BASE[@]}" "$@"
}

container_name() {
    if [[ -n "${SERVE_JOB_NAME}" ]]; then
        printf 'astrai-server-%s\n' "${SERVE_JOB_NAME}"
    else
        printf 'astrai-server\n'
    fi
}

service_name() {
    if [[ "${SERVE_GPU_ENABLED:-true}" == "false" ]]; then
        printf 'server-cpu\n'
    else
        printf 'server\n'
    fi
}

set_profile_args() {
    PROFILE_ARGS=()
    if [[ "${SERVE_GPU_ENABLED:-true}" == "false" ]]; then
        PROFILE_ARGS=(--profile cpu)
    fi
}

init_environment() {
    mkdir -p "${SERVE_PARAM_DIR}"
    log_info "Model: ${SERVE_PARAM_DIR}"
}

preflight() {
    require_command docker
    docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
    [[ -d "${SERVE_PARAM_DIR}" ]] || die "Model directory not found: ${SERVE_PARAM_DIR}"
    [[ -s "${SERVE_PARAM_DIR}/config.json" ]] ||
        die "Model config not found: ${SERVE_PARAM_DIR}/config.json"
    [[ -s "${SERVE_PARAM_DIR}/model.safetensors" ]] ||
        die "Model weights not found: ${SERVE_PARAM_DIR}/model.safetensors"

    if [[ "${SERVE_GPU_ENABLED}" == "false" ]] && [[ "${SERVE_DEVICE}" != "cpu" ]]; then
        die "runtime.gpu.enabled is false but server.device is '${SERVE_DEVICE}'; use server.device: cpu"
    fi

    compose config --quiet
    log_info "Preflight passed (service: $(service_name), device: ${SERVE_DEVICE})"
}

runtime_environment_args() {
    RUNTIME_ENV_ARGS=()
    local pair
    while IFS= read -r -d '' pair; do
        RUNTIME_ENV_ARGS+=(--env "${pair}")
    done < <(python3 "${ROOT_DIR}/scripts/tools/serve_runtime.py" environment "${CONFIG_FILE}")
}

start_server() {
    local foreground="$1"
    shift
    local container running
    local -a run_options
    preflight
    runtime_environment_args
    set_profile_args
    container="$(container_name)"
    running="$(docker inspect --format '{{.State.Running}}' "${container}" 2>/dev/null || true)"
    [[ "${running}" != "true" ]] || die "Server is already running: ${container}"
    docker rm "${container}" >/dev/null 2>&1 || true

    run_options=(
        --volume "${CONFIG_FILE}:/run/astrai/serve.yaml:ro"
        "${RUNTIME_ENV_ARGS[@]}"
    )
    if [[ "${foreground}" == "true" ]]; then
        compose "${PROFILE_ARGS[@]}" run --build --rm --service-ports \
            "${run_options[@]}" "$(service_name)" \
            python -m scripts.tools.server --config /run/astrai/serve.yaml "$@"
    else
        compose "${PROFILE_ARGS[@]}" run -d --build --service-ports \
            --name "${container}" "${run_options[@]}" "$(service_name)" \
            python -m scripts.tools.server --config /run/astrai/serve.yaml "$@"
        log_info "Server started; run scripts/serve.sh logs ${CONFIG_FILE} to follow it"
    fi
}

stop_server() {
    local container
    container="$(container_name)"
    docker stop --timeout 30 "${container}" >/dev/null 2>&1 ||
        log_warn "Server container is not running"
    docker rm "${container}" >/dev/null 2>&1 || true
}

show_status() {
    docker ps -a --filter "name=^/$(container_name)$"
}

main() {
    local command="${1:-}" config="${SERVE_CONFIG_FILE:-${ROOT_DIR}/serve.yaml}"
    [[ -n "${command}" ]] || { usage; exit 1; }
    shift || true

    if [[ "${command}" =~ ^(help|-h|--help)$ ]]; then
        usage
        return
    fi

    if [[ $# -gt 0 && "$1" != --* ]]; then
        config="$1"
        shift
    fi
    load_config "${config}"

    case "${command}" in
        init) init_environment ;;
        preflight) preflight ;;
        build)
            set_profile_args
            preflight
            compose "${PROFILE_ARGS[@]}" build "$(service_name)"
            ;;
        up) start_server false "$@" ;;
        run) start_server true "$@" ;;
        down) stop_server ;;
        restart) stop_server; start_server false ;;
        logs) docker logs -f --tail "${SERVE_LOG_TAIL:-200}" "$(container_name)" ;;
        status) show_status ;;
        *) die "Unknown command: ${command}" ;;
    esac
}

main "$@"
