"""Unit tests for the serving CLI YAML merge logic."""

import click
import pytest
import torch
from click.testing import CliRunner

from scripts.tools.server import (
    _merge_yaml_into_kwargs,
    _resolve_server_config,
    server_command,
)


def _passed() -> dict:
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,
        "param_path": None,
        "device": "cuda",
        "dtype": "bfloat16",
        "max_batch_size": 16,
        "max_seq_len": None,
    }


def test_yaml_overrides_click_defaults_but_not_explicit_cli(tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "server:\n  device: cpu\n  dtype: float16\n  max_batch_size: 8\n",
        encoding="utf-8",
    )
    merged = _merge_yaml_into_kwargs(
        str(config_path), _passed(), explicit_keys={"device"}
    )

    assert merged["device"] == "cuda"
    assert merged["dtype"] == "float16"
    assert merged["max_batch_size"] == 8


def test_resolve_config_yaml_wins_by_default(tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "server:\n  port: 9000\n  max_seq_len: 2048\n",
        encoding="utf-8",
    )
    resolved = _resolve_server_config(str(config_path), _passed())

    assert resolved["port"] == 9000
    assert resolved["max_seq_len"] == 2048
    assert resolved["device"] == "cuda"
    assert resolved["dtype"] == "bfloat16"


def test_resolve_config_rejects_bad_dtype(tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text("server:\n  dtype: fp8\n", encoding="utf-8")

    with pytest.raises(click.UsageError, match="server.dtype"):
        _resolve_server_config(str(config_path), _passed())


def test_server_command_rejects_bad_yaml_dtype(tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text("server:\n  dtype: fp8\n", encoding="utf-8")

    result = CliRunner().invoke(server_command, ["--config", str(config_path)])

    assert result.exit_code == 2
    assert "server.dtype" in result.output


def test_server_command_merges_yaml_and_cli(tmp_path, monkeypatch):
    """Full CLI path: YAML values apply, explicit CLI flags override, args reach run_server."""
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "server:\n  device: cpu\n  dtype: float16\n  max_batch_size: 8\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run_server(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("scripts.tools.server.run_server", fake_run_server)
    result = CliRunner().invoke(
        server_command,
        ["--config", str(config_path), "--max_batch_size", "32"],
    )

    assert result.exit_code == 0, result.output
    assert captured["device"] == "cpu"
    assert captured["dtype"] == torch.float16
    assert captured["max_batch_size"] == 32
    assert captured["port"] == 8000


def test_config_option_rejects_missing_file(tmp_path):
    result = CliRunner().invoke(
        server_command, ["--config", str(tmp_path / "nope.yaml")]
    )

    assert result.exit_code == 2
