from pathlib import Path

import click
import torch
import yaml
from click.core import ParameterSource

from astrai.inference import run_server

_DTYPES = ["bfloat16", "float16", "float32"]
_SERVER_KEYS = (
    "host",
    "port",
    "reload",
    "param_path",
    "device",
    "dtype",
    "max_batch_size",
    "max_seq_len",
)


def _merge_yaml_into_kwargs(
    config_path: str,
    passed_kwargs: dict,
    explicit_keys: set[str] | None = None,
) -> dict:
    """Merge Click defaults, YAML server values, then explicit CLI values."""
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise click.UsageError(f"Serving config must be a mapping: {config_path}")
    server = config.get("server") or {}
    if not isinstance(server, dict):
        raise click.UsageError("top-level server section must be a mapping")

    unknown = sorted(set(server) - set(_SERVER_KEYS))
    if unknown:
        click.echo(
            f"Warning: ignoring unknown server config keys: {', '.join(unknown)}",
            err=True,
        )

    merged = dict(passed_kwargs)
    merged.update({key: server[key] for key in _SERVER_KEYS if key in server})
    if explicit_keys is None:
        explicit_keys = set(passed_kwargs)
    for key in explicit_keys:
        if key in passed_kwargs:
            merged[key] = passed_kwargs[key]
    return merged


def _as_int(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise click.UsageError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise click.UsageError(f"{name} must be an integer, got {value!r}") from None


def _resolve_server_config(
    config_path: str,
    passed_kwargs: dict,
    explicit_keys: set[str] | None = None,
) -> dict:
    """Merge YAML values, then coerce and validate the resolved settings.

    ``explicit_keys`` are CLI flags that win over YAML; when None, YAML values
    win over Click defaults.
    """
    merged = _merge_yaml_into_kwargs(config_path, passed_kwargs, explicit_keys or set())
    resolved = dict(merged)
    resolved["port"] = _as_int(resolved["port"], "server.port") or 8000
    resolved["max_batch_size"] = (
        _as_int(resolved["max_batch_size"], "server.max_batch_size") or 16
    )
    resolved["max_seq_len"] = _as_int(resolved["max_seq_len"], "server.max_seq_len")
    resolved["reload"] = bool(resolved["reload"])
    if resolved["dtype"] not in _DTYPES:
        raise click.UsageError(
            f"server.dtype must be one of {', '.join(_DTYPES)}, got {resolved['dtype']!r}"
        )
    return resolved


@click.command(name="serve", help="Launch inference server (OpenAI-compatible API).")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Serving YAML config. CLI flags override YAML values.",
)
@click.option("--host", default="0.0.0.0", help="Host address.")
@click.option("--port", type=int, default=8000, help="Port number.")
@click.option(
    "--reload", is_flag=True, default=False, help="Enable auto-reload for development."
)
@click.option(
    "--param_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to model parameters.",
)
@click.option("--device", default="cuda", help="Device to load model on.")
@click.option(
    "--dtype",
    type=click.Choice(_DTYPES),
    default="bfloat16",
    help="Data type for model weights.",
)
@click.option(
    "--max_batch_size",
    type=int,
    default=16,
    help="Maximum batch size for continuous batching.",
)
@click.option(
    "--max_seq_len",
    type=int,
    default=None,
    help="Maximum sequence length (KV cache size + prompt truncation). Uses model config if not set.",
)
@click.pass_context
def server_command(
    ctx,
    config_path,
    host,
    port,
    reload,
    param_path,
    device,
    dtype,
    max_batch_size,
    max_seq_len,
):
    """Launch inference server (OpenAI-compatible API)."""
    if config_path:
        passed_kwargs = {
            "host": host,
            "port": port,
            "reload": reload,
            "param_path": param_path,
            "device": device,
            "dtype": dtype,
            "max_batch_size": max_batch_size,
            "max_seq_len": max_seq_len,
        }
        explicit_keys = {
            key
            for key in passed_kwargs
            if ctx.get_parameter_source(key) is ParameterSource.COMMANDLINE
        }
        resolved = _resolve_server_config(config_path, passed_kwargs, explicit_keys)
        host = resolved["host"]
        port = resolved["port"]
        reload = resolved["reload"]
        param_path = resolved["param_path"]
        device = resolved["device"]
        dtype = resolved["dtype"]
        max_batch_size = resolved["max_batch_size"]
        max_seq_len = resolved["max_seq_len"]
        click.echo(f"Config: {config_path}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    project_root = Path(__file__).parent.parent.parent
    param_path = param_path or str(project_root / "params")

    click.echo(f"Starting server on http://{host}:{port}")
    click.echo(f"Model: {param_path}  |  Device: {device}  |  Dtype: {dtype}")
    run_server(
        host=host,
        port=port,
        reload=reload,
        device=device,
        dtype=dtype_map[dtype],
        param_path=Path(param_path),
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )


if __name__ == "__main__":
    server_command()
