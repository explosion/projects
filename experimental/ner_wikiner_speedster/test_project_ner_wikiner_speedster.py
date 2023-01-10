from pathlib import Path

from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets


def test_wikiner_speedster_project():
    overrides = {
        "vars.optimize_opts": "-ot constrained",
        "vars.corpora_dev_limit": 10,
        "vars.corpora_train_limit": 10,
    }
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", overrides=overrides, capture=True)
