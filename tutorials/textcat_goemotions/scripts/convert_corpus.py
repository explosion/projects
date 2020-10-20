from pathlib import Path
import typer
from spacy.tokens import DocBin
import spacy


ASSETS_DIR = Path(__file__).parent.parent / "assets"
CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def read_categories(path: Path):
    return path.open().read().strip().split("\n")


def read_tsv(file_):
    for line in file_:
        text, labels, annotator = line.split("\t")
        yield {
            "text": text,
            "labels": [int(label) for label in labels.split(",")],
            "annotator": annotator
        }


def convert_record(nlp, record, categories):
    """Convert a record from the tsv into a spaCy Doc object."""
    doc = nlp.make_doc(record["text"])
    # All categories other than the true ones get value 0
    doc.cats = {category: 0 for category in categories}
    # True labels get value 1
    for label in record["labels"]:
        doc.cats[categories[label]] = 1
    return doc


def main(assets_dir: Path=ASSETS_DIR, corpus_dir: Path=CORPUS_DIR, lang: str="en"):
    """Convert the GoEmotion corpus's tsv files to spaCy's binary format."""
    categories = read_categories(assets_dir / "categories.txt")
    nlp = spacy.blank(lang)
    for tsv_file in assets_dir.iterdir():
        if not tsv_file.parts[-1].endswith(".tsv"):
            continue
        records = read_tsv(tsv_file.open(encoding="utf8"))
        docs = [convert_record(nlp, record, categories) for record in records]
        out_file = corpus_dir / tsv_file.with_suffix(".spacy").parts[-1]
        out_data = DocBin(docs=docs).to_bytes()
        with out_file.open("wb") as file_:
            file_.write(out_data)


if __name__ == "__main__":
    typer.run(main)
