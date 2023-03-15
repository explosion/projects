import random
from pathlib import Path

import typer
from spacy.tokens import DocBin
from spacy.util import get_lang_class
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def view_spans(
    # fmt: off
    input_path: Path = Arg(..., help="Input spaCy file", exists=True),
    lang: str = Opt("xx", "--lang", "-l", help="Language for the vocab"),
    display_size: int = Opt(3, "--size", "-sz", help="Number of Doc.spans to show"),
    shuffle: bool = Opt(False, "--shuffle", "-s", help="Shuffle the items")
    # fmt: on
):
    """Helper function to check if spans were saved correctly"""
    nlp = get_lang_class(lang)()
    db = DocBin().from_disk(input_path)
    docs = list(db.get_docs(nlp.vocab))
    msg.info(f"Found {len(docs)} docs in {str(input_path)}")
    msg.info(f"Showing {display_size} docs")

    if shuffle:
        random.shuffle(docs)
    for doc in docs[:display_size]:
        msg.divider()
        msg.text(doc.text)
        msg.text(doc.spans, color="yellow")


if __name__ == "__main__":
    typer.run(view_spans)
