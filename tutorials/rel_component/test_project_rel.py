import pytest
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


@pytest.mark.xfail(reason='Requires spaCy v3.0.0rc4')
def test_rel_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all")
