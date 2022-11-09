import sys
from pathlib import Path

import pytest
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets


@pytest.mark.skipif(sys.platform == "win32", reason="nebullvm does support windows devices yet")
def test_wikiner_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", capture=True)
