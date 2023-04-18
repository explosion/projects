from pathlib import Path
from typing import Dict, List, Optional, Set

import spacy
import srsly
import typer
from spacy import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def get_labels(docs: List[str]) -> Set[str]:
    """Get all entity types"""
    ents = []
    ents.extend([ent.label_ for doc in docs for ent in doc.ents])
    return set(ents)


def convert_record(nlp: Language, record: Dict[str, str]) -> Doc:
    """Convert a record from the OpenAI output into a spaCy Doc object"""
    doc = nlp.make_doc(record.get("text"))
    spans = [
        doc.char_span(
            start_idx=span.get("start"),
            end_idx=span.get("end"),
            label=span.get("label").capitalize(),
        )
        for span in record.get("spans", [])
    ]
    doc.set_ents(spans)
    return doc


def filter_by_members(on: List[Doc], by: List[str]) -> List[Doc]:
    """Filter documents based on membership"""
    docs = [doc for doc in on if doc.text in set(by)]
    return docs


def read_records(path: Path) -> List[Doc]:
    nlp = spacy.blank("en")
    if path.suffix == ".spacy":
        doc_bin = DocBin().from_disk(path)
        docs = list(doc_bin.get_docs(nlp.vocab))
    elif path.suffix == ".jsonl":
        docs = [convert_record(nlp, record) for record in srsly.read_jsonl(path)]
    else:
        msg.fail(
            f"Unknown file extension: {path.suffix}. Pass a spaCy or JSONL file.",
            exits=1,
        )
    return docs


def evaluate(
    # fmt: off
    prediction_path: Path = Arg(..., help="Path to the predictions"),
    reference_path: Path = Arg(..., help="Path to the gold annotations"),
    output_path: Optional[Path] = Opt(None, "--output-path", "--output", "-o", help="Path to save the metrics"),
    # fmt: on
):
    """Evaluate the zero-shot annotations of GPT-3"""
    # Read records
    reference = read_records(reference_path)
    predicted = read_records(prediction_path)

    # When we're downsampling, it's possible that not all labels
    # will be present in the predicted annotations.
    assert get_labels(predicted).issubset(get_labels(reference))
    examples = [Example(pred, ref) for pred, ref in zip(predicted, reference)]

    # Perform evaluation
    scores = Scorer.score_spans(examples, attr="ents")
    msg.text(title="Scores", text=scores)
    if output_path:
        msg.good(f"Saved metrics to {output_path}")
        srsly.write_json(output_path, scores)


if __name__ == "__main__":
    typer.run(evaluate)
