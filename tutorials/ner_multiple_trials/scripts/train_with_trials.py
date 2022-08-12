import typer

import random
from typing import Optional, Dict, Any, Union
from pathlib import Path
from wasabi import msg
import typer
import logging
import sys


from spacy.cli._util import Arg, Opt, parse_config_overrides, show_validation_error
from spacy.cli._util import import_code, setup_gpu
from spacy.training.loop import train as train_nlp
from spacy.training.initialize import init_nlp
from spacy import util

app = typer.Typer()

MAX_SEED = 2**32 - 1


@app.command(
    "train", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train_cli(
    # fmt: off
    ctx: typer.Context,  # This is only used to read additional arguments
    config_path: Path = Arg(..., help="Path to config file", exists=True, allow_dash=True),
    num_trials: int = Opt(5, "--n-trials", "-n", help="Number of trials to run"),
    output_path: Optional[Path] = Opt(None, "--output", "--output-path", "-o", help="Output directory to store trained pipeline in"),
    code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU")
    # fmt: on
):
    """spaCy train command with multiple trials"""
    util.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)

    seeds = [random.randint(0, MAX_SEED) for _ in range(num_trials)]

    for trial, seed in enumerate(seeds):
        msg.divider(f"Performing trial {trial+1} of {num_trials} (seed={seed})")
        overrides["system.seed"] = seed
        train(
            config_path,
            output_path / str(seed),
            num_trials=num_trials,
            use_gpu=use_gpu,
            overrides=overrides,
        )

def train(
    config_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    use_gpu: int = -1,
    overrides: Dict[str, Any] = util.SimpleFrozenDict(),
):
    config_path = util.ensure_path(config_path)
    output_path = util.ensure_path(output_path)
    # Make sure all files and paths exists if they are needed
    if not config_path or (str(config_path) != "-" and not config_path.exists()):
        msg.fail("Config file not found", config_path, exits=1)
    if not output_path:
        msg.info("No output directory provided")
    else:
        if not output_path.exists():
            output_path.mkdir(parents=True)
            msg.good(f"Created output directory: {output_path}")
        msg.info(f"Saving to output directory: {output_path}")
    setup_gpu(use_gpu)
    with show_validation_error(config_path):
        config = util.load_config(config_path, overrides=overrides, interpolate=False)
    msg.divider("Initializing pipeline")
    with show_validation_error(config_path, hint_fill=False):
        nlp = init_nlp(config, use_gpu=use_gpu)
    msg.good("Initialized pipeline")
    msg.divider("Training pipeline")
    train_nlp(nlp, output_path, use_gpu=use_gpu, stdout=sys.stdout, stderr=sys.stderr)



if __name__ == "__main__":
    app()
