from pathlib import Path
from typing import Dict, List, Optional, Set

import spacy
import srsly
import typer
from spacy import Language
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def get_labels(docs: List[str]) -> Set[str]:
    """Get all entity types"""
    ents = []
    ents.extend([ent.label_ for doc in docs for ent in doc.ents])
    return set(ents)


def convert_record(nlp: Language, record: Dict[str, str]):
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


def evaluate_gpt(
    # fmt: off
    input_path: Path = Arg(..., help="Path to the JSONL predictions of GPT-3"),
    test_path: Path = Arg(..., help="Path to the test set in spaCy format"),
    output_path: Optional[Path] = Opt(None, "--output-path", "--output", "-o", help="Path to save the metrics"),
    # fmt: on
):
    """Evaluate the zero-shot annotations of GPT-3"""
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(test_path)

    # Create examples for evaluation
    reference = list(doc_bin.get_docs(nlp.vocab))
    predicted = [convert_record(nlp, pred) for pred in srsly.read_jsonl(input_path)]
    assert get_labels(reference) == get_labels(predicted)
    examples = [Example(pred, ref) for pred, ref in zip(predicted, reference)]

    # Perform evaluation
    scores = Scorer.score_spans(examples, attr="ents")
    msg.text(title="Scores", text=scores)
    if output_path:
        msg.good(f"Saved metrics to {output_path}")
        srsly.write_json(output_path, scores)


if __name__ == "__main__":
    typer.run(evaluate_gpt)
