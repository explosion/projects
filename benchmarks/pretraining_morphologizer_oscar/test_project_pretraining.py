from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def test_project_pretraining():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "data")
    project_run(root, "training_char", overrides={"vars.epochs": 1})
