from pathlib import Path

from util import get_tok2vecs, check_tok2vecs, has_listener
from util import check_pipeline_names, get_listeners

import typer
import spacy
from spacy.language import Language
from wasabi import msg

def create_resume_config(base_model: str, output_path: Path):
    """Given an input pipeline, produce a config for resuming training.

    A config for resuming training is the same as the input config, but with
    all components sourced.
    """

    nlp = spacy.load(base_model)
    conf = nlp.config

    for comp in nlp.pipe_names:
        conf["components"][comp] = {"source": base_model}

    conf.to_disk(output_path)

if __name__ == "__main__":

    app = typer.Typer(name="Resume Config Creator")
    app.command("resume_config", no_args_is_help=True)(create_resume_config)
    app()
