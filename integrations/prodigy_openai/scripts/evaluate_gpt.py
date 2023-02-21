from pathlib import Path
from typing import Dict, Optional

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

# The Prodigy recipe normalizes the labels into lowercase for easier parsing so
# we need to map them with the proper capitalization.
CATEGORY_MAP = {
    "argument_for": "Argument_for",
    "noargument": "NoArgument",
    "argument_against": "Argument_against",
}


def convert_record(
    nlp: Language, record: Dict[str, str], category_map: Dict[str, str] = CATEGORY_MAP
):
    """Convert a record from the OpenAI output into a spaCy Doc object"""
    doc = nlp.make_doc(record.get("text"))
    label = category_map.get(record.get("accept")[0])
    doc.cats = {category: 0 for category in category_map.values()}
    doc.cats[label] = 1
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
    examples = [Example(pred, ref) for pred, ref in zip(predicted, reference)]

    # Perform evaluation
    scores = Scorer.score_cats(
        examples, attr="cats", labels=CATEGORY_MAP.values(), multi_label=False
    )
    msg.text(title="Scores", text=scores)
    if output_path:
        msg.good(f"Saved metrics to {output_path}")
        srsly.write_json(output_path, scores)


if __name__ == "__main__":
    typer.run(evaluate_gpt)
