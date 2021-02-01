from spacy.cli.project.run import project_run
from pathlib import Path


@pytest.mark.skip(
    reason="CLI is acting up - fixing it but skipping the test in the meantime."
)
def test_benchmark_speed_project():
    root = Path(__file__).parent
    # project_assets(root)   # there are currently no assets defined for this project
    project_run(root, "setup")
    project_run(root, "timing_cpu")
