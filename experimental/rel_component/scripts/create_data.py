import typer
from spacy.lang.en import English
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc
from wasabi import Printer

msg = Printer()


def main(train_file: Path, dev_file: Path, test_file: Path):
    """Creating some toy data.
        In an realistic application, this should be read and parsed from some corpus."""
    nlp = English()
    Doc.set_extension("rel", default={})
    _create_train_data(nlp, train_file)
    _create_dev_data(nlp, dev_file)
    _create_test_data(nlp, test_file)


def _create_train_data(nlp, data_file: Path):
    # there can be relation multiple labels for two entities
    doc1 = nlp("Amsterdam is the capital of the Netherlands.")
    doc1.ents = [Span(doc1, 0, 1, label="LOC"), Span(doc1, 6, 7, label="LOC")]
    doc1._.rel = {(0, 6): {"CAPITAL_OF": 1.0, "ALLY": 1.0}, (6, 0): {"ALLY": 1.0}}

    # with many entities close to eachother, the number of relations will grow exponentially
    doc2 = nlp("I like Ghent, London and Berlin")
    doc2.ents = [
        Span(doc2, 2, 3, label="LOC"),
        Span(doc2, 4, 5, label="LOC"),
        Span(doc2, 6, 7, label="LOC"),
    ]
    doc2._.rel = {
        (2, 4): {"UNRELATED": 1.0},
        (4, 6): {"UNRELATED": 1.0},
        (2, 6): {"UNRELATED": 1.0},
        (4, 2): {"UNRELATED": 1.0},
        (6, 4): {"UNRELATED": 1.0},
        (6, 2): {"UNRELATED": 1.0},
    }

    # there can be missing data
    doc3 = nlp("The United Kingdom and the US have made trade agreements.")
    doc3.ents = [Span(doc3, 1, 3, label="LOC"), Span(doc3, 5, 6, label="LOC")]
    doc3._.rel = {(1, 5): {"ALLY": 1.0}}

    # docs = [doc1, doc2, doc3]
    docs = [doc1, doc2]
    docbin = DocBin(docs=docs, store_user_data=True)
    docbin.to_disk(data_file)
    msg.info(f"Wrote {len(docs)} training articles to {data_file}")


def _create_dev_data(nlp, data_file: Path):
    doc1 = nlp("Brussels is the capital of Belgium. I like Paris and New York.")
    doc1.ents = [
        Span(doc1, 0, 1, label="LOC"),
        Span(doc1, 5, 6, label="LOC"),
        Span(doc1, 9, 10, label="LOC"),
        Span(doc1, 11, 13, label="LOC"),
    ]
    doc1._.rel = {
        (0, 5): {"CAPITAL_OF": 1.0, "ALLY": 1.0},
        (5, 0): {"ALLY": 1.0},
        (9, 11): {"UNRELATED": 1.0},
        (11, 9): {"UNRELATED": 1.0},
    }

    doc2 = nlp("I like Ghent, London and Berlin")
    doc2.ents = [
        Span(doc2, 2, 3, label="LOC"),
        Span(doc2, 4, 5, label="LOC"),
        Span(doc2, 6, 7, label="LOC"),
    ]
    doc2._.rel = {
        (2, 4): {"UNRELATED": 1.0},
        (4, 6): {"UNRELATED": 1.0},
        (2, 6): {"UNRELATED": 1.0},
        (4, 2): {"UNRELATED": 1.0},
        (6, 4): {"UNRELATED": 1.0},
        (6, 2): {"UNRELATED": 1.0},
    }

    #docs = [doc1, doc2]
    docs = [doc1]
    docbin = DocBin(docs=docs, store_user_data=True)
    docbin.to_disk(data_file)
    msg.info(f"Wrote {len(docs)} dev articles to {data_file}")


def _create_test_data(nlp, data_file: Path):
    """Unseen, unannotated data"""
    texts = ["Amsterdam is the capital of the Netherlands.",
             "I like Ghent, London and Berlin",
             "The United Kingdom and the US have made trade agreements.",
             "Brussels is the capital of Belgium. I like Paris and New York.",
             "London is the capital of the United Kingdom; the capital of Belgium is Brussels.",
             ]
    docs = list(nlp.pipe(texts))
    docbin = DocBin(docs=docs, store_user_data=True)
    docbin.to_disk(data_file)
    msg.info(f"Wrote {len(docs)} test articles to {data_file}")


if __name__ == "__main__":
    typer.run(main)
