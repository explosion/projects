from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def test_experimental_coref():
    root = Path(__file__).parent
    # provide a placeholder value here, since we won't have real OntoNotes
    overrides = {"vars.ontonotes": "assets", "vars.gpu_id": -1}
    project_assets(root, overrides=overrides, extra=True)
    project_run(root, "ci-test", overrides=overrides)
