from pathlib import Path

from spacy.cli.project.assets import project_assets
from spacy.cli.project.run import project_run


def test_span_labeling_datasets():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", capture=True)
