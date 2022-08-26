""" Testing all project steps. """

from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def test_nel_emerson_project():
    # Running the entire project takes too long. Alternative?
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "download_mewsli-9", capture=True)
    project_run(root, "preprocess", capture=True)
    project_run(root, "download_model", capture=True)
    # todo run entire workflow once filtered dumps are available
    # project_run(root, "parse_wiki_dumps --use_filtered_dumps", capture=True)
    # project_run(root, "create_kb", capture=True)
    # project_run(root, "compile_corpora", capture=True)
    # project_run(root, "train", capture=True)
    # project_run(root, "evaluate", capture=True)
