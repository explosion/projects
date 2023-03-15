import logging
import random
from pathlib import Path
from typing import Optional

import typer
from spacy import util
from spacy.cli._util import import_code, parse_config_overrides
from spacy.cli.train import train
from wasabi import msg

app = typer.Typer()

MAX_SEED = 2**32 - 1


@app.command(
    "train", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train_cli(
    # fmt: off
    ctx: typer.Context,  # This is only used to read additional arguments
    config_path: Path = typer.Argument(..., help="Path to config file", exists=True, allow_dash=True),
    num_trials: int = typer.Option(5, "--n-trials", "-n", help="Number of trials to run"),
    output_path: Optional[Path] = typer.Option(None, "--output", "--output-path", "-o", help="Output directory to store trained pipeline in"),
    code_path: Optional[Path] = typer.Option(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    verbose: bool = typer.Option(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
    use_gpu: int = typer.Option(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU")
    # fmt: on
):
    """spaCy train command with multiple trials"""
    util.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)

    seeds = [random.randint(0, MAX_SEED) for _ in range(num_trials)]

    for trial, seed in enumerate(seeds):
        msg.divider(f"Performing trial {trial+1} of {num_trials} (seed={seed})")
        overrides["training.seed"] = seed
        train(
            config_path,
            output_path / f"trial_{trial}",
            use_gpu=use_gpu,
            overrides=overrides,
        )


if __name__ == "__main__":
    app()
