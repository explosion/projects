from collections import Counter
from pathlib import Path
from typing import List

import typer
from wasabi import msg

import spacy
from spacy.tokens import DocBin

Arg = typer.Argument
Opt = typer.Option


def get_distribution(
    # fmt: off
    input_path: List[Path] = typer.Argument(..., help="Path to the spaCy file."), 
    n: int = typer.Option(5, "-n", "--top-n", help="Top-n entities to include in the report."),
    # fmt: on
):
    """Get the distribution of entities given a list of spaCy files"""
    nlp = spacy.blank("en")

    docs = []
    for path in input_path:
        doc_bin = DocBin().from_disk(path)
        _docs = list(doc_bin.get_docs(nlp.vocab))
        docs.extend(_docs)

    # Get the entity counts
    num_docs = len(docs)
    msg.info(f"Found {num_docs} documents in {', '.join([str(p) for p in input_path])}")
    entity_counts = Counter()
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ not in entity_counts:
                entity_counts[ent.label_] = 0
            else:
                entity_counts[ent.label_] += 1

    # Get the distribution (normalize everything)
    total = sum(entity_counts.values())
    _fmt_counts = " ".join(
        [
            f"{ent} ({(count / total) * 100:.2f}%)"
            for ent, count in entity_counts.most_common(n)
        ]
    )
    msg.text(f"Top-{n} entities by count: {_fmt_counts}")


if __name__ == "__main__":
    typer.run(get_distribution)
