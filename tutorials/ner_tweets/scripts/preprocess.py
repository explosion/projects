import json
from pathlib import Path

import spacy
import typer
from spacy.tokens import DocBin
from wasabi import msg
from tqdm import tqdm


ASSETS_DIR = Path(__file__).parent.parent / "assets"

app = typer.Typer()


@app.command()
def prepare_train_data(assets_dir: Path = ASSETS_DIR, model: str = "en_core_web_lg"):
    """Convert raw text file of tweets into spaCy's binary format

    At the onset, we will already use the `en_core_web_lg` model to
    store the detected entities in the DocBin.

    assets_dir (Path): path to the assets directory for saving
    model (str): the spaCy pretrained model to use
    """
    nlp = spacy.load(model)
    raw_text_fp = assets_dir / "train.txt"

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


@app.command()
def prepare_test_data(assets_dir: Path = ASSETS_DIR):
    """Convert raw text file into jsonl as Prodigy input

    Using the test.txt causes some weird line breaks, probably due to the hashtag.
    Standardizing them into Prodigy's preferred input format is much better.

    assets_dir (Path): path to the assets directory for saving
    """

    raw_text_fp = assets_dir / "test.txt"
    records = raw_text_fp.read_text().strip().split("\n")
    msg.text(f"Converting {len(records)} records to JSONL format...")

    out_file = assets_dir / raw_text_fp.with_suffix(".jsonl").parts[-1]

    with out_file.open("w") as fp:
        for record in records:
            entry = {"text": record}
            json.dump(entry, fp)
            fp.write("\n")


@app.command()
def prepare_datasets():
    """Prepare both training and test datasets to suitable formats"""
    prepare_train_data()
    prepare_test_data()


if __name__ == "__main__":
    app()
