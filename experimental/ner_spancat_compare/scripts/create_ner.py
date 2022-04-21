from pathlib import Path
from typing import Dict, List, Optional
from copy import copy

import spacy
import typer
from spacy.tokens import Doc, DocBin, Span
from spacy.util import filter_spans
from wasabi import msg


def get_all_labels(docs: List[Doc], spans_key="sc") -> List[str]:
    labels = []
    for doc in docs:
        spans = doc.spans[spans_key]
        for span in spans:
            if span.label_ not in labels:
                labels.append(span.label_)

    return labels


def create_ner_dataset(
    docs: List[Doc],
    labels: List[str],
    spans_key: str = "sc",
) -> Dict[str, List[Doc]]:
    docs_per_label = {}
    for label in labels:
        doc_with_ents = []
        for doc in docs:
            # It seems that without copying, it maintains
            # a reference to the doc, overwriting the ents inside.
            _doc = copy(doc)
            spans = _doc.spans[spans_key]
            spans_single_label = []
            for span in spans:
                if span.label_ == label:
                    spans_single_label.append(span)
            try:
                _doc.ents = spans_single_label
            except ValueError:
                # It means that there are *still* overlapping entities
                _doc.ents = filter_spans(spans_single_label)
            doc_with_ents.append(_doc)
        docs_per_label[label] = doc_with_ents
    return docs_per_label


def main(
    train: Path = typer.Option(..., "--train", exists=True),
    dev: Path = typer.Option(..., "--dev", exists=True),
    test: Path = typer.Option(..., "--test", exists=True),
    outdir: Optional[Path] = typer.Option(..., "-o", "--output-dir", exists=True),
):

    nlp = spacy.blank("en")

    train_docs = list(DocBin().from_disk(train).get_docs(nlp.vocab))
    dev_docs = list(DocBin().from_disk(dev).get_docs(nlp.vocab))
    test_docs = list(DocBin().from_disk(test).get_docs(nlp.vocab))

    # We only need the train docs because the test and dev labels should
    # be a subset of the train labels.
    labels = get_all_labels(train_docs)
    msg.info(f"Found labels: {labels}")

    with msg.loading(f"Splitting the dataset per label..."):
        train_dict = create_ner_dataset(train_docs, labels=labels)
        dev_dict = create_ner_dataset(dev_docs, labels=labels)
        test_dict = create_ner_dataset(test_docs, labels=labels)
        msg.good("Datasets were now split according to label!")

    if outdir:
        with msg.loading(f"Saving datasets to {outdir}..."):
            for label in labels:
                DocBin(docs=train_dict[label]).to_disk(outdir / f"train_{label}.spacy")
                DocBin(docs=dev_dict[label]).to_disk(outdir / f"dev_{label}.spacy")
                DocBin(docs=test_dict[label]).to_disk(outdir / f"test_{label}.spacy")
        msg.good(f"Saved spaCy files to {outdir}!")


if __name__ == "__main__":
    typer.run(main)
