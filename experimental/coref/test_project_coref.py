import os
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def test_experimental_coref():

    # This is set globally in testing to make sure tests are fast, but can't be
    # used for this project because passing training-specific parameters breaks
    # the spacy assemble command.
    os.environ["SPACY_CONFIG_OVERRIDES"] = ""
    root = Path(__file__).parent
    # provide a placeholder value here, since we won't have real OntoNotes
    overrides = {
        "vars.ontonotes": "assets",
        "vars.gpu_id": -1,
        "vars.max_epochs": 10,
        "vars.config_dir": "configs/test",
    }
    project_assets(root, overrides=overrides, extra=True)
    project_run(root, "ci-test", overrides=overrides)
