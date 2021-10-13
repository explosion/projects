from pathlib import Path
from typing import List

import spacy
import typer
from sklearn.model_selection import train_test_split
from skweak.aggregation import HMM
from spacy.tokens import Doc, DocBin
from wasabi import msg

from .weak_supervision import UnifiedNERAnnotator


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    model_output_path: Path = typer.Argument(..., dir_okay=False),
    training_output_path: Path = typer.Argument(..., dir_okay=False),
    eval_output_path: Path = typer.Argument(..., dir_okay=False),
    train_size: float = 0.8,
    weak_supervision: bool = True,
):
    """Augment a dataset and split it into training and eval

    input_path (Path): the training dataset to augment
    model_output_path (Path): path to save the HMM model for weak supervision
    training_output_path (Path): path to save the output training data
    eval_output_path (Path): path to save the output evaluation data
    """
    msg.info(f"Reading data from {input_path}")
    db = DocBin()
    nlp = spacy.blank("en")
    docs = list(db.from_disk(input_path).get_docs(nlp.vocab))

    # Perform data augmentation
    if weak_supervision:
        msg.info("Performing weak supervision...")
        docs = augment_with_weak_supervision(docs, model_output_path)

    # Split the dataset based on ratio
    train_data, eval_data = train_test_split(docs, train_size=train_size)

    # Save training and eval datasets
    db_train = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    for doc in train_data:
        db_train.add(doc)
    db_train.to_disk(training_output_path)
    msg.good(f"Saved train data to disk! (size={len(train_data)})")

    db_dev = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    for doc in eval_data:
        db_dev.add(doc)
    db_dev.to_disk(eval_output_path)
    msg.good(f"Saved eval data to disk! (size={len(eval_data)})")


def augment_with_weak_supervision(
    docs: List[Doc], model_output_path: Path
) -> List[Doc]:
    """Perform augmentation via weak supervision"""
    unified_annotator = UnifiedNERAnnotator()
    unified_annotator.add_all_annotators()
    # Annotate our training set using all annotators
    msg.text("Labelling dataset with all annotators...")
    docs_annotated = unified_annotator.pipe(docs)
    # Fit a hidden markov model based on the outputs
    msg.text("Fitting a hidden markov model...")
    label_model = HMM("hmm", ["PERSON"])
    label_model.add_underspecified_label("ENT", ["PERSON"])
    label_model.fit(docs_annotated)
    label_model.save(model_output_path)
    msg.good(f"Model saved to {model_output_path}")
    # Annotate again using the learned model
    return list(label_model.pipe(docs))


if __name__ == "__main__":
    typer.run(main)
