from pathlib import Path

import spacy
import typer
from spacy.tokens import DocBin
from wasabi import msg


ASSETS_DIR = Path(__file__).parent.parent / "assets"


def convert_to_spacy(assets_dir: Path = ASSETS_DIR, lang: str = "en"):
    """Convert raw text file of tweets into spaCy's binary format"""
    nlp = spacy.blank(lang)
    raw_text_fp = assets_dir / "raw.txt"

    msg.text(f"Converting records to spacy format...")
    records = raw_text_fp.read_text().strip().split("\n")
    docs = [nlp.make_doc(record) for record in records]

    out_file = assets_dir / raw_text_fp.with_suffix(".spacy").parts[-1]
    out_data = DocBin(docs=docs).to_bytes()
    with out_file.open("wb") as fp:
        fp.write(out_data)
    msg.good(f"Done! Saved to {out_file}")


if __name__ == "__main__":
    typer.run(convert_to_spacy)
