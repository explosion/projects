from pathlib import Path


ASSETS_PATH = Path(__file__).parent.parent / "assets"

# Source model / data for annotators
BTC_MODEL_PATH = ASSETS_PATH / "btc"
WIKIDATA_PATH = ASSETS_PATH / "wikidata_tokenised.json"
CRUNCHBASE_PATH = ASSETS_PATH / "crunchbase.json"
NAMES_PATH = ASSETS_PATH / "first_names.json"

NAME_PREFIXES = [
    "-",
    "von",
    "van",
    "de",
    "di",
    "le",
    "la",
    "het",
    "'t'",
    "dem",
    "der",
    "den",
    "d'",
    "ter",
]
