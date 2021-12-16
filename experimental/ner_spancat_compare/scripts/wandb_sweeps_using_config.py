from pathlib import Path

import typer
from typer.models import Required
import wandb
from spacy import util
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from thinc.api import Config


def main(
    default_config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to the default spaCy configuration.",
    ),
    output_path: Path = typer.Argument(
        ...,
        exists=False,
        dir_okay=True,
        help="Output directory to save the trained spaCy model.",
    ),
    train_path: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help="Override the paths.train parameter with this value.",
        prompt_required=True,
    ),
    dev_path: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help="Override the paths.dev parameter with this value.",
        prompt_required=True,
    ),
):
    """Modified train command with W&B's Sweeps

    This command merges WandB's Sweep config with our default
    spaCy config. Useful for hyperparameter search.
    """
    loaded_local_config = util.load_config(default_config)
    breakpoint()
    with wandb.init() as run:
        sweeps_config = Config(util.dot_to_dict(run.config))
        merged_config = Config(loaded_local_config).merge(sweeps_config)
        nlp = init_nlp(merged_config)
        train(nlp, output_path, use_gpu=True)


if __name__ == "__main__":
    typer.run(main)
