from pathlib import Path
from typing import List

import spacy
import typer
from sklearn.model_selection import train_test_split
from skweak.aggregation import HMM
from skweak.utils import docbin_writer
from spacy.tokens import Doc, DocBin
from wasabi import msg

from .weak_supervision import UnifiedNERAnnotator


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    model_output_path: Path = typer.Argument(..., dir_okay=False),
    training_output_path: Path = typer.Argument(..., dir_okay=False),
    dev_output_path: Path = typer.Argument(..., dir_okay=False),
    train_size: float = 0.8,
    weak_supervision: bool = True,
    limit: int = typer.Option(None, envvar="NUM_DOCS_TO_AUGMENT"),
):
    """Augment a dataset and split it into training and eval

    input_path (Path): the training dataset to augment
    model_output_path (Path): path to save the HMM model for weak supervision
    training_output_path (Path): path to save the output training data
    dev_output_path (Path): path to save the output evaluation data
    """
    msg.info(f"Reading data from {input_path}")
    db = DocBin()
    nlp = spacy.blank("en")
    docs = list(db.from_disk(input_path).get_docs(nlp.vocab))

    # Perform data augmentation
    if weak_supervision:
        msg.info("Performing weak supervision...")
        docs = augment_weak_supervision(docs, model_output_path, limit)

    # Split the dataset based on ratio
    train_data, dev_data = train_test_split(docs, train_size=train_size)

    # Save training and eval datasets
    serialize_docs(train_data, training_output_path)
    serialize_docs(dev_data, dev_output_path)


def serialize_docs(docs: List[Doc], output_path: Path, span_name: str = "hmm"):
    """Serialize the annotated documents into the spaCy format

    docs (List[Doc]): list of Doc objects to serialize
    output_path (Path): path to save the serialized dataset
    span_name (str): name of the span to include as entities
    """
    for doc in docs:
        try:
            doc.ents = doc.spans[span_name]
        except KeyError:
            pass
    docbin_writer(docs, str(output_path))
    msg.good(f"Saved data to disk! (size={len(docs)})")


def augment_weak_supervision(
    docs: List[Doc], model_output_path: Path, limit: int
) -> List[Doc]:
    """Perform augmentation via weak supervision

    This step first collates all the labelling functions found in UnifiedNERAnnotator,
    then trains a hidden-markov model to estimate a unified annotator.

    docs (List[Doc]): list of Doc objects to augment
    model_output_path (Path): path to save the Hidden Markov Model (HMM)
    limit (int): optional, number of documents to annotate, useful for testing
    """
    unified_annotator = UnifiedNERAnnotator().add_all_annotators()
    msg.info(f"Total number of annotators: {len(unified_annotator.annotators)}")
    # Annotate our training set using all annotators
    msg.text("Labelling dataset with all annotators...")
    docs_to_annotate = docs[:limit]
    docs_annotated = list(unified_annotator.pipe(docs_to_annotate))
    # Fit a hidden markov model based on the outputs
    msg.text("Fitting a hidden markov model...")
    label_model = HMM("hmm", ["PERSON"])
    label_model.add_underspecified_label("ENT", ["PERSON"])
    label_model.fit(docs_annotated)
    label_model.save(model_output_path)
    msg.good(f"Model saved to {model_output_path}")
    # Annotate again using the learned model
    docs_annotated_hmm = list(label_model.pipe(docs))
    return docs_annotated_hmm


if __name__ == "__main__":
    typer.run(main)
