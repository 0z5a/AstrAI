"""CLI: JSONL → tokenized .h5/.bin via config-driven Pipeline."""

import click

from astrai import setup_logging
from astrai.config.preprocess_config import PipelineConfig
from astrai.preprocessing.pipeline import Pipeline


@click.command(
    name="preprocess", help="Tokenize and pack raw JSONL data into .bin/.h5 format."
)
@click.argument("inputs", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--output_dir", "-o", type=click.Path(), required=True, help="Output directory."
)
@click.option(
    "--config",
    "-c",
    "pipeline_config",
    type=click.Path(exists=True),
    required=True,
    help="Pipeline config JSON.",
)
@click.option(
    "--tokenizer_path",
    type=click.Path(exists=True),
    default="params",
    help="Path to tokenizer directory.",
)
@click.option("--batch_size", type=int, default=None, help="Records per batch.")
def preprocess_command(inputs, output_dir, pipeline_config, tokenizer_path, batch_size):
    """Tokenize and pack raw JSONL data into .bin/.h5 format."""
    config = PipelineConfig.from_file(pipeline_config)
    if batch_size is not None:
        if batch_size < 1:
            raise click.BadParameter("--batch_size must be at least 1")
        config.preprocessing.batch_size = batch_size

    click.echo(f"Preprocessing {len(inputs)} file(s) → {output_dir}")
    Pipeline(
        config=config,
        input_paths=list(inputs),
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
    ).run()
    click.echo("Done.")


if __name__ == "__main__":
    setup_logging()
    preprocess_command()
