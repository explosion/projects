import sys
from pathlib import Path

import pytest
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets


@pytest.mark.skipif(sys.platform == "win32", reason="nebullvm does support windows devices yet")
def test_wikiner_project():
    import openvino_telemetry as tm
    tm.Telemetry.opt_out("UA-17808594-29")
    overrides = {
        "vars.optimize_opts": "-ot constrained",
        "vars.corpora_dev_limit": 10,
        "vars.corpora_train_limit": 10,
    }
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", overrides=overrides, capture=True)
