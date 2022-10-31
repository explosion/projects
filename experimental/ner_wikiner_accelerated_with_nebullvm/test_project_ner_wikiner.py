import sys
from pathlib import Path

from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets


def test_wikiner_project():
    if sys.platform.startswith('win'):
        # skip test on windows since nebullvm is not supporting windows
        #  devices yet. 
        return
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", capture=True)
