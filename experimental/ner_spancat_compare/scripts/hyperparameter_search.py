from pathlib import Path

import typer
from typer.models import Required
import wandb
from spacy import util
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from thinc.api import Config
from wasabi import msg

from .constants import Pipeline


SWEEP_CONFIG_NER = {
    "method": "random",
    "metric": {
        "name": "ents_f",  # https://spacy.io/api/scorer#score
        "goal": "maximize",
    },
    "parameters": {
        "training.dropout": {
            "distribution": "uniform",
            "max": 0.5,
            "min": 0.05,
        },
    },
}


SWEEP_CONFIG_SPANCAT = {"method": "random"}


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
    pipeline: Pipeline = typer.Option(
        Pipeline.spancat,
        show_default=True,
        help="Pipeline to search parameters from. The Sweep config will depend on this value.",
    ),
):
    """Modified train command with W&B's Sweeps

    This command merges WandB's Sweep config with our default
    spaCy config. Useful for hyperparameter search.
    """

    def train_spacy():
        loaded_local_config = util.load_config(default_config)
        loaded_local_config["paths"]["train"] = train_path
        loaded_local_config["paths"]["dev"] = dev_path
        with wandb.init() as run:
            sweeps_config = Config(util.dot_to_dict(run.config))
            merged_config = Config(loaded_local_config).merge(sweeps_config)
            nlp = init_nlp(merged_config)
            train(nlp, output_path, use_gpu=True)

    if pipeline == "spancat":
        sweep_config = SWEEP_CONFIG_SPANCAT
    elif pipeline == "ner":
        sweep_config = SWEEP_CONFIG_NER
    else:
        msg.fail(f"Unknown pipeline value: {pipeline}")

    sweep_id = wandb.sweep(sweep_config, project="ner_spancat_compare")
    wandb.agent(sweep_id, train_spacy, count=20)


if __name__ == "__main__":
    typer.run(main)
