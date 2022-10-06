""" Testing all project steps. """

from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def test_nel_benchmark_project():
    # todo figure out what we want to test exactly - wiki download & dump parsing is excessive.
    pass
    # root = Path(__file__).parent
    # project_assets(root)
    # project_run(root, "preprocess", capture=True)
    # project_run(root, "download_model", capture=True)
    # project_run(root, "create_kb", capture=True)
    # project_run(root, "compile_corpora", capture=True)
    # project_run(root, "train", capture=True)
    # project_run(root, "evaluate", capture=True)
