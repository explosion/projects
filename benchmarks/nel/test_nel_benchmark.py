""" Testing all project steps. """
import os

import pytest
from pathlib import Path
import sys
from spacy.cli.project.run import project_run


@pytest.mark.skipif(sys.platform == "win32", reason="Skipping on Windows (for now) due to platform-specific scripts.")
def test_nel_benchmark():
    overrides_key = "SPACY_CONFIG_OVERRIDES"
    root = Path(__file__).parent
    project_run(root, "download_mewsli9", capture=True)
    project_run(root, "download_model", capture=True)
    project_run(root, "wikid_clone", capture=True)
    project_run(root, "preprocess", capture=True)
    # Temporarily disable override env variables, since these may result in config validation errors in this
    # project-in-project situation.
    overrides = os.environ.pop(overrides_key, None)
    project_run(root, "wikid_download_assets", capture=True)
    project_run(root, "wikid_parse", capture=True)
    project_run(root, "wikid_create_kb", capture=True)
    # Re-enable config overrides, if set before.
    if overrides:
        os.environ[overrides_key] = overrides
    project_run(root, "extract_annotations", capture=True)
    project_run(root, "compile_corpora", capture=True)
    project_run(root, "retrieve_mentions_candidates", capture=True)
    project_run(root, "train", capture=True, overrides={"vars.training_max_steps": 1, "vars.training_max_epochs": 1})
    project_run(root, "evaluate", capture=True)
    project_run(root, "compare_evaluations", capture=True)
