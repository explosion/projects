"""Convert annotation from spaCy v2 TRAIN_DATA format to spaCy v3 .spacy
format."""
import srsly
import typer
import warnings
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example


def convert(lang: str, input_path: Path, output_path: Path):
    nlp = spacy.blank(lang)
    db = DocBin()
    for text, annot in srsly.read_json(input_path):
        example = Example.from_dict(nlp.make_doc(text), annot)
        db.add(example.reference)
    db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(convert)
