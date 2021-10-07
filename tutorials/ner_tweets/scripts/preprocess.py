from pathlib import Path

import spacy
import typer
from spacy.tokens import DocBin
from wasabi import msg
from tqdm import tqdm


ASSETS_DIR = Path(__file__).parent.parent / "assets"


def convert_to_spacy(assets_dir: Path = ASSETS_DIR, model: str = "en_core_web_lg"):
    """Convert raw text file of tweets into spaCy's binary format

    At the onset, we will already use the `en_core_web_lg` model to
    store the detected entities in the DocBin.

    assets_dir (Path): path to the assets directory for saving
    model (str): the spaCy pretrained model to use
    """
    nlp = spacy.load(model)
    raw_text_fp = assets_dir / "raw.txt"

    records = raw_text_fp.read_text().strip().split("\n")
    msg.text(f"Converting {len(records)} records to spacy format...")
    with nlp.select_pipes(enable="ner"):  # we only need the NER component
        msg.text(f"Using pipes: {nlp.pipe_names}")
        docs = [nlp(record) for record in tqdm(records)]

    out_file = assets_dir / raw_text_fp.with_suffix(".spacy").parts[-1]
    out_data = DocBin(docs=docs).to_bytes()
    with out_file.open("wb") as fp:
        fp.write(out_data)
    msg.good(f"Done! Saved to {out_file}")


if __name__ == "__main__":
    typer.run(convert_to_spacy)
