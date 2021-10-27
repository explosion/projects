import os
from pathlib import Path

from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets


def test_ner_tweets_project():
    root = Path(__file__).parent
    os.environ["NUM_DOCS_TO_AUGMENT"] = "10"  # augmentation takes time, so we limit it
    project_assets(root)
    project_run(root, "all", capture=True)
