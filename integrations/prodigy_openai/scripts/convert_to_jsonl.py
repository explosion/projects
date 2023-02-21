from pathlib import Path

import spacy
import srsly
import typer
from spacy.tokens import DocBin
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def convert_to_jsonl(
    # fmt: off
    input_path: Path = Arg(..., help="Path to the spaCy file to convert."),
    output_path: Path = Arg(..., help="Filepath to save the converted JSONL output."),
    lang: str = Opt("en", "--lang", "-l", help="Language code of the corpus."),
    # fmt: on
):
    """Convert spaCy file into JSONL to use for Prodigy"""

    nlp = spacy.blank(lang)
    doc_bin = DocBin().from_disk(input_path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    texts = [{"text": doc.text} for doc in docs]
    msg.text(f"Found {len(texts)} documents in {input_path}")

    srsly.write_jsonl(output_path, texts)
    msg.good(f"Saved texts to {output_path}")


if __name__ == "__main__":
    typer.run(convert_to_jsonl)
