import pytest
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


@pytest.mark.skip(reason="TODO: reenable")
def test_taggerparser_ud_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all")
