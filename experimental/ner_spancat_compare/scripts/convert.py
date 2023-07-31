from pathlib import Path
from typing import List

import typer
from spacy.tokens import Doc, DocBin, Span, SpanGroup
from spacy.training.converters import conll_ner_to_docs
from wasabi import msg

from .constants import DOC_DELIMITER


def convert_iob_to_docs(
    data: str,
    num_levels: int = 4,
    spans_key: str = "sc",
    doc_delimiter: str = DOC_DELIMITER,
) -> List[Doc]:
    """Parse GENIA dataset into spaCy docs

    Our strategy here is to reuse the conll -> ner method from spaCy and
    re-apply that n times. We don't want to write our own ConLL/IOB parser.
    """
    docs = data.split("\n\n")
    iob_per_level = []

    for level in range(num_levels):
        doc_list = []
        for doc in docs:
            tokens = [t for t in doc.split("\n") if t]
            token_list = []
            for token in tokens:
                annot = token.split("\t")
                # First element is always the token text
                text = annot[0]
                label = annot[level + 1]
                _token = " ".join([text, label])
                token_list.append(_token)
            doc_list.append("\n".join(token_list))
        annotations = doc_delimiter.join(doc_list)
        iob_per_level.append(annotations)

    # We then copy all the entities from doc.ents into
    # doc.spans later on. But first, let's have a "canonical" docs
    # to copy into
    docs_per_level = [list(conll_ner_to_docs(iob)) for iob in iob_per_level]
    docs_with_spans: List[Doc] = []

    for docs in zip(*docs_per_level):
        doc = docs[0]
        # recreate all spans for the same underlying doc
        spans = []
        for span in [ent for doc in docs for ent in doc.ents]:
            spans.append(Span(doc, span.start, span.end, span.label_))
        group = SpanGroup(doc, name=spans_key, spans=spans)
        doc.spans[spans_key] = group
        docs_with_spans.append(doc)

    return docs_with_spans


def main(
    filepath: Path = typer.Argument(..., exists=True),
    output: Path = typer.Option(..., "-o", "--output", exists=False),
):
    with filepath.open("r", encoding="utf-8") as f:
        data = f.read()

    docs = convert_iob_to_docs(data)
    with msg.loading(f"Saving into DocBin..."):
        doc_bin = DocBin(docs=docs)
        doc_bin.to_disk(output)
        msg.good(f"Saved to {output}")


if __name__ == "__main__":
    typer.run(main)
