from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


import pytest
@pytest.mark.skip(reason="temp")
def test_rel_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all")
