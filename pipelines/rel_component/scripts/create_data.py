import typer
from spacy.lang.en import English
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc


def main(data_file: Path):
    """Creating some toy data.
    In an realistic application, this should be read and parsed from  some corpus."""
    nlp = English()
    Doc.set_extension("rel", default={})

    doc1 = nlp("Amsterdam is the capital of the Netherlands.")
    doc1.ents = [Span(doc1, 0, 1, label="LOC"),
                Span(doc1, 6, 7, label="LOC")]
    doc1._.rel = {(0,6): "CAPITAL_OF"}

    doc2 = nlp("I like Ghent and Berlin.")
    doc2.ents = [Span(doc2, 2, 3, label="LOC"),
                Span(doc2, 4, 5, label="LOC")]
    doc2._.rel = {(2, 4): "UNRELATED", (4,2): "UNRELATED"}

    doc3 = nlp("The United Kingdom and the US have made trade agreements.")
    doc3.ents = [Span(doc3, 1, 3, label="LOC"),
                Span(doc3, 5, 6, label="LOC")]
    doc3._.rel = {(1, 5): "ALLY_OF", (5, 1): "ALLY_OF"}

    docbin = DocBin(docs=[doc1, doc2, doc3], store_user_data=True)
    docbin.to_disk(data_file)


if __name__ == "__main__":
    typer.run(main)