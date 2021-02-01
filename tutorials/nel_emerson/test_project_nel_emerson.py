import pytest
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


@pytest.mark.skip(
    reason="CLI is acting up - fixing it but skipping the test in the meantime."
)
def test_nel_emerson_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "download")
    project_run(root, "training")
