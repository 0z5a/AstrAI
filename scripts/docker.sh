#!/bin/bash
set -e

IMAGE_NAME="${ASTRAI_IMAGE:-astrai}"
IMAGE_TAG="${ASTRAI_TAG:-latest}"
PORT="8000"
GPU=true
RUN_ARGS=()

usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
  build              Build the image
  run [--] [ARGS]    Run a container; ARGS after -- are passed to the container

Options:
  --gpu              Enable GPU support (default)
  --no-gpu           Disable GPU support
  --port PORT        Host port for run (default: 8000)
  -h, --help         Show this help

Environment:
  ASTRAI_IMAGE       Image name (default: astrai)
  ASTRAI_TAG         Image tag (default: latest)

Examples:
  $0 build
  $0 run
  $0 run --port 8080 -- python -m scripts.tools.server --port 8000 --device cuda
EOF
}

build_image() {
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
}

run_container() {
    local gpu_args=()
    [ "$GPU" = true ] && gpu_args=(--gpus all)
    docker run "${gpu_args[@]}" -p "${PORT}:8000" "${IMAGE_NAME}:${IMAGE_TAG}" "$@"
}

main() {
    local command=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            build|run)
                command="$1"
                shift
                ;;
            --gpu)
                GPU=true
                shift
                ;;
            --no-gpu)
                GPU=false
                shift
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --)
                shift
                RUN_ARGS=("$@")
                break
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    case "$command" in
        build)
            build_image
            ;;
        run)
            run_container "${RUN_ARGS[@]}"
            ;;
        *)
            echo "No command specified. Use --help for usage" >&2
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
