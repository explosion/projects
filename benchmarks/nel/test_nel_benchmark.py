""" Testing all project steps. """
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
    project_run(root, "wikid_download_assets", capture=True)
    project_run(root, "wikid_parse", capture=True)
    project_run(root, "wikid_create_kb", capture=True)
    project_run(root, "parse_corpus", capture=True)
    project_run(root, "compile_corpora", capture=True)
    project_run(root, "train", capture=True, overrides={"vars.training_max_steps": 1})
    project_run(root, "evaluate", capture=True)
    project_run(root, "compare_evaluations", capture=True)
