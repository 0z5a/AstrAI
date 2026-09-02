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
        "kv_cache_tokens": None,
        "kv_cache_page_size": 1,
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


@pytest.mark.parametrize(
    ("yaml_body", "message"),
    [
        ("kv_cache_tokens: 0", "server.kv_cache_tokens must be positive"),
        ("kv_cache_page_size: 0", "server.kv_cache_page_size must be positive"),
        (
            "kv_cache_page_size: 64",
            "server.kv_cache_page_size requires server.kv_cache_tokens",
        ),
        (
            "kv_cache_tokens: 100\n  kv_cache_page_size: 64",
            "server.kv_cache_tokens must be divisible",
        ),
    ],
)
def test_resolve_config_rejects_invalid_kv_cache_settings(tmp_path, yaml_body, message):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(f"server:\n  {yaml_body}\n", encoding="utf-8")

    with pytest.raises(click.UsageError, match=message):
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
        "server:\n"
        "  device: cpu\n"
        "  dtype: float16\n"
        "  max_batch_size: 8\n"
        "  kv_cache_tokens: 4096\n"
        "  kv_cache_page_size: 64\n",
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
    assert captured["kv_cache_tokens"] == 4096
    assert captured["kv_cache_page_size"] == 64


def test_server_command_rejects_invalid_cli_cache_settings():
    result = CliRunner().invoke(
        server_command,
        ["--kv_cache_tokens", "100", "--kv_cache_page_size", "64"],
    )

    assert result.exit_code == 2
    assert "server.kv_cache_tokens must be divisible" in result.output


def test_config_option_rejects_missing_file(tmp_path):
    result = CliRunner().invoke(
        server_command, ["--config", str(tmp_path / "nope.yaml")]
    )

    assert result.exit_code == 2
