from pathlib import Path

SPAN_KEY = "sc"


class Directories:
    ROOT_DIR = Path(__file__).parent.parent
    # Main raw data directories
    ASSETS_DIR = ROOT_DIR / "assets"
    DATA_DIR = ASSETS_DIR / "ebm_nlp_1_00"
    # Context and text directories
    TEXT_DIR = DATA_DIR / "documents"
    # Labels and annotations directories
    ANNOTATIONS_DIR = DATA_DIR / "annotations" / "aggregated" / "starting_spans"
    PARTICIPANTS_DIR = ANNOTATIONS_DIR / "participants"
    INTERVENTIONS_DIR = ANNOTATIONS_DIR / "interventions"
    OUTCOMES_DIR = ANNOTATIONS_DIR / "outcomes"
