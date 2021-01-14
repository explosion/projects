import typer
from pathlib import Path
from spacy.training.loop import train
from spacy.training.initialize import init_nlp
from spacy.util import load_config


def main(config_dir: Path, results_dir: Path):
    """Run all config files instead of the default one.
    Ideally, these runs are parellellized instead of run in sequence."""
    if not config_dir.exists() or not config_dir.is_dir():
        print(f"Could not read from folder {config_dir}")
    for config_path in config_dir.iterdir():
        print(f"Training on {config_path}")
        config = load_config(config_path, interpolate=False)
        nlp = init_nlp(config, use_gpu=False)
        output_path = results_dir / config_path.stem
        if not output_path.exists():
            output_path.mkdir()
        train(nlp, output_path, use_gpu=False)


if __name__ == "__main__":
    typer.run(main)
