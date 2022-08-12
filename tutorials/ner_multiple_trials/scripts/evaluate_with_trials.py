from pathlib import Path
from typing import Optional, Iterable
from enum import Enum

import typer
from spacy.cli._util import Arg, Opt, import_code
from spacy.cli.evaluate import evaluate

app = typer.Typer()


class ModelType(str, Enum):
    model_best = "model-best"
    model_last = "model-last"


@app.command("evaluate")
def evaluate_cli(
    # fmt: off
    training_dir_path: Path = Arg(..., help="Model name or path"),
    data_path: Path = Arg(..., help="Location of binary evaluation data in .spacy format", exists=True),
    output_dir: Optional[Path] = Opt(None, "--output-dir", "-o", help="Output directory to store the JSON file for metrics", dir_okay=True),
    code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
    gold_preproc: bool = Opt(False, "--gold-preproc", "-G", help="Use gold preprocessing"),
    displacy_path: Optional[Path] = Opt(None, "--displacy-path", "-dp", help="Directory to output rendered parses as HTML", exists=True, file_okay=False),
    displacy_limit: int = Opt(25, "--displacy-limit", "-dl", help="Limit of parses to render as HTML"),
    model_type: ModelType = Opt(ModelType.model_best, "--model-type", "-t", help="Whether to evaluate on model-best or model-last.")
    # fmt: on
):
    """spaCy evaluate command with multiple trials

    This script only demonstrates how we can successively evaluate multiple models
    in a given directory.
    """
    import_code(code_path)
    models = (p / model_type.value for p in training_dir_path.iterdir())
    for model in models:
        _, seed, _ = model.parts
        evaluate(
            model,
            data_path,
            output=output_dir / f"{seed}.json",
            use_gpu=use_gpu,
            gold_preproc=gold_preproc,
            displacy_path=displacy_path,
            displacy_limit=displacy_limit,
            silent=False,
        )


if __name__ == "__main__":
    app()
