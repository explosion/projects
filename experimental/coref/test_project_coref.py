import os
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def test_experimental_coref():

    # This is set globally in testing to make sure tests are fast, but can't be
    # used for this project because passing training-specific parameters breaks
    # the spacy assemble command. We need to restore the old value after this
    # test though.
    old_overrides = os.environ.get("SPACY_CONFIG_OVERRIDES", "")
    os.environ["SPACY_CONFIG_OVERRIDES"] = ""

    root = Path(__file__).parent
    # provide a placeholder value here, since we won't have real OntoNotes
    overrides = {
        "vars.ontonotes": "assets",
        "vars.gpu_id": -1,
        "vars.max_epochs": 10,
        "vars.config_dir": "configs/test",
    }
    try:
        project_assets(root, overrides=overrides, extra=True)
        for step in ("prep-artificial-unit-test-data", "train", "eval"):
            project_run(root, step, overrides=overrides)
    except:
        raise
    finally:
        os.environ["SPACY_CONFIG_OVERRIDES"] = old_overrides
