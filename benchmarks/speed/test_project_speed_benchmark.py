from spacy.cli.project.run import project_run
from pathlib import Path


def test_benchmark_speed_project():
    root = Path(__file__).parent
    # project_assets(root)   # there are currently no assets defined for this project
    project_run(root, "setup")
    project_run(root, "timing_cpu")
