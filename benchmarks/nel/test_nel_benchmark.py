""" Testing all project steps. """
import os

import pytest
from pathlib import Path
import sys
from spacy.cli.project.run import project_run


@pytest.mark.skipif(sys.platform == "win32", reason="Skipping on Windows (for now) due to platform-specific scripts.")
def test_nel_benchmark():
    root = Path(__file__).parent
    project_run(root, "download_mewsli9", capture=True)
    project_run(root, "download_model", capture=True)
    project_run(root, "wikid_clone", capture=True)
    project_run(root, "preprocess", capture=True)
    overrides_key = "SPACY_CONFIG_OVERRIDES"
    overrides = os.environ.pop(overrides_key) if overrides_key in os.environ else None
    project_run(root, "wikid_download_assets", capture=True)
    project_run(root, "wikid_parse", capture=True)
    project_run(root, "wikid_create_kb", capture=True)
    if overrides:
        os.environ[overrides_key] = overrides
    project_run(root, "parse_corpus", capture=True)
    project_run(root, "compile_corpora", capture=True)
    project_run(root, "train", capture=True)
    project_run(root, "evaluate", capture=True)
    project_run(root, "compare_evaluations", capture=True)
