from pathlib import Path

import click
import torch

from astrai import setup_logging
from astrai.inference import run_server

_DTYPES = ["bfloat16", "float16", "float32"]


@click.command(name="serve", help="Launch inference server (OpenAI-compatible API).")
@click.option("--host", default="0.0.0.0", help="Host address.")
@click.option("--port", type=int, default=8000, help="Port number.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
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
def server_command(
    host, port, reload, param_path, device, dtype, max_batch_size, max_seq_len
):
    """Launch inference server (OpenAI-compatible API)."""
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
    setup_logging()
    server_command()
