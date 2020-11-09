from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def test_benchmark_textcat_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "install")
    project_run(root, "all")
