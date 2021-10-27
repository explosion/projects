from pathlib import Path


ASSETS_PATH = Path(__file__).parent.parent / "assets"

# Source model / data for annotators
BTC_MODEL_PATH = ASSETS_PATH / "data" / "btc"
WIKIDATA_PATH = ASSETS_PATH / "wikidata_tokenised.json"
CRUNCHBASE_PATH = ASSETS_PATH / "crunchbase.json"
NAMES_PATH = ASSETS_PATH / "first_names.json"

# Taken from skweak's data utilities
# https://github.com/NorskRegnesentral/skweak/blob/670fcdec680930ce3e497886d06d61e6a1f2c195/examples/ner/data_utils.py
NAME_PREFIXES = [
    "-",
    "von",
    "van",
    "de",
    "di",
    "le",
    "la",
    "het",
    "'t",
    "dem",
    "der",
    "den",
    "d'",
    "ter",
]
