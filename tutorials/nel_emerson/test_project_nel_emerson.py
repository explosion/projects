from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path
import pytest


@pytest.mark.skip(reason="Not ready yet")
def test_project():
    root = Path(__file__).parent
    project_run(root, "setup")
    project_assets(root)
    project_run(root, "all")
