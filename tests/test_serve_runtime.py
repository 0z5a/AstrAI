"""Unit tests for the serving runtime configuration parser."""

import pytest

from scripts.tools.serve_runtime import load_runtime


def _write(tmp_path, body: str) -> str:
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(body, encoding="utf-8")
    return str(config_path)


def test_runtime_exports_defaults(tmp_path):
    config_path = _write(
        tmp_path,
        "runtime:\n"
        "  port: 8000\n"
        "  paths:\n"
        "    param: ./params\n"
        "server:\n"
        "  device: cuda\n",
    )
    runtime = load_runtime(config_path)
    assert runtime["SERVE_PORT"] == "8000"
    assert runtime["SERVE_CONTAINER_PORT"] == "8000"
    assert runtime["SERVE_PARAM_DIR"] == str((tmp_path / "params").resolve())
    assert runtime["SERVE_GPU_ENABLED"] == "true"
    assert "CUDA_VISIBLE_DEVICES" not in runtime
    assert runtime["SERVE_DEVICE"] == "cuda"
    assert runtime["CUDA_TAG"] == "cu128"
    assert runtime["SERVE_JOB_NAME"] == ""


def test_runtime_gpu_disabled_requires_cpu(tmp_path):
    config_path = _write(
        tmp_path,
        "runtime:\n  gpu:\n    enabled: false\nserver:\n  device: cuda\n",
    )
    with pytest.raises(ValueError, match="server.device must be 'cpu'"):
        load_runtime(config_path)


def test_runtime_gpu_devices_single_and_ports(tmp_path):
    config_path = _write(
        tmp_path,
        "runtime:\n"
        "  gpu:\n"
        "    devices: [1]\n"
        "  port: 8080\n"
        "server:\n"
        "  port: 9000\n"
        "  device: cuda\n",
    )
    runtime = load_runtime(config_path)
    assert runtime["SERVE_PORT"] == "8080"
    assert runtime["SERVE_CONTAINER_PORT"] == "9000"
    assert runtime["CUDA_VISIBLE_DEVICES"] == "1"


def test_runtime_gpu_devices_rejects_multi(tmp_path):
    config_path = _write(
        tmp_path,
        "runtime:\n  gpu:\n    devices: [0, 1]\n",
    )
    with pytest.raises(ValueError, match="single-device"):
        load_runtime(config_path)


def test_runtime_port_out_of_range(tmp_path):
    config_path = _write(tmp_path, "runtime:\n  port: 70000\n")
    with pytest.raises(ValueError, match="between 1 and 65535"):
        load_runtime(config_path)


def test_runtime_environment_export(tmp_path):
    config_path = _write(
        tmp_path,
        "runtime:\n  environment:\n    TOKENIZERS_PARALLELISM: 'false'\n",
    )
    runtime = load_runtime(config_path)
    assert runtime["environment"] == {"TOKENIZERS_PARALLELISM": "false"}
