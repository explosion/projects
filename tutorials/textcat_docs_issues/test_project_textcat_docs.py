from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def test_textcat_issues_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", capture=True)
