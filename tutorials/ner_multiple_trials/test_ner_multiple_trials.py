from pathlib import Path

from spacy.cli.project.assets import project_assets
from spacy.cli.project.run import project_run


def test_ner_multiple_trials_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(
        root,
        "all",
        overrides={
            "vars.trials": 2,
            "vars.config": "ner_efficiency.cfg",
            "vars.max_steps": 200,
            "vars.limit": 100,
        },
        capture=True
    )
