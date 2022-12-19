from pathlib import Path

from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets

def test_spanruler_restaurant_reviews():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", capture=True)