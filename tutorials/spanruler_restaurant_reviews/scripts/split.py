import random
from pathlib import Path
from typing import Optional

import typer
import spacy
from spacy.tokens import DocBin
from wasabi import msg


Arg = typer.Argument
Opt = typer.Option

LANG = "en"


def split(
    # fmt: off
    input_path: Path = Arg(..., help="Path to the train-dev spaCy file."),
    output_dir: Path = Arg(..., help="Output path to store training and dev files."),
    train_size: float = Opt(0.8, help="Size of the training set."),
    seed: Optional[int] = Opt(None, help="Random seed for shuffling. If None, no shuffling is done."),
    # fmt: on
):
    """Split the dataset into training and dev partitions.

    The original MIT Restaurant reviews dataset doesn't provide splits for the
    training and dev data. Hence, we will split it ourselves.
    """
    nlp = spacy.blank(LANG)

    traindev_doc_bin = DocBin().from_disk(input_path)
    traindev_docs = list(traindev_doc_bin.get_docs(nlp.vocab))
    msg.text(f"Found {len(traindev_docs)} documents.")

    if seed is not None:
        msg.text(f"Shuffling the documents using seed '{seed}'.")
        random.seed(seed)
        random.shuffle(traindev_docs)

    num_train = int(len(traindev_docs) * train_size)
    train_docs = traindev_docs[:num_train]
    dev_docs = traindev_docs[num_train:]

    msg.text(
        f"Done splitting documents ({len(traindev_docs)}) into "
        f"train ({len(train_docs)}) and dev ({len(dev_docs)})."
    )

    train_doc_bin = DocBin(docs=train_docs)
    dev_doc_bin = DocBin(docs=dev_docs)

    train_doc_bin.to_disk(output_dir / "train.spacy")
    dev_doc_bin.to_disk(output_dir / "dev.spacy")
    msg.good(f"Saved files to '{output_dir}' directory.")


if __name__ == "__main__":
    typer.run(split)
