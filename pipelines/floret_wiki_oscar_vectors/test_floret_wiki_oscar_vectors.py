import pytest
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def test_floret_web_vectors():
    root = Path(__file__).parent
    overrides = {
        "vars.lang": "yo",
        "vars.n_process_tokenize": 2,
        "vars.vector_thread": 2,
        "vars.downloaded_dir": "./temp",
        "vars.extracted_dir": "./temp",
        "vars.tokenized_dir": "./temp",
        "vars.vector_input_dir": "./temp",
    }
    project_assets(root, overrides=overrides)
    project_run(root, "all", overrides=overrides, capture=True)
