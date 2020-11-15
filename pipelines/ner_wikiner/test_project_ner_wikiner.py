import pytest
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


@pytest.mark.skip(reason="TODO: numpy error on Windows")
def test_wikiner_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all")
