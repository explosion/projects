from pathlib import Path

from spacy.cli.project.assets import project_assets
from spacy.cli.project.run import project_run


def test_ud_tagalog_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", overrides={"vars.n_folds": 1, "vars.gpu": 0})
