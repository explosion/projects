"""Split a spaCy-formatted file into train, dev, and test partitions"""

from pathlib import Path
from typing import Tuple, Optional, Any, List, Sequence

import random
import typer
import spacy
from math import ceil
from spacy.tokens import DocBin
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def _train_dev_test_split(
    data: Sequence[Any],
    train_size,
    dev_size: float,
    test_size: float,
    shuffle: Optional[bool] = False,
    seed: Optional[int] = None,
) -> Tuple[List[Any], List[Any], List[Any]]:
    if shuffle:
        if not seed:
            raise ValueError("Must provide 'seed' when 'shuffle = True'")
        rng = random.Random(seed)
        rng.shuffle(data)
    n_samples = len(data)
    n_test = ceil(test_size * n_samples)
    n_dev = ceil(dev_size * n_samples)
    n_train = n_samples - (n_test + n_dev)
    train = data[:n_train]
    dev = data[n_train:n_train+n_dev]
    test = data[n_train+n_dev:]
    return train, dev, test


def split_docs(
    # fmt: off
    input_path: Path,
    output_dir: Path,
    split_size: Tuple[float, float, float] = Arg((0.8, 0.1, 0.1), help="Split sizes for train/dev/test respectively"),
    shuffle: bool = Opt(False, "--shuffle", "-sf", help="Shuffle the dataset before splitting"),
    seed: Optional[int] = Opt(None, "--seed", "-sd", help="Random seed for shuffling the data")
    # fmt: on
):
    if sum(split_size) != 1.0:
        msg.fail(
            "Split sizes for train, dev, and test should sum up to 1.0 "
            f"({' + '.join(map(str, split_size))} != 1.0)",
            exits=1,
        )

    nlp = spacy.blank("xx")
    db = DocBin().from_disk(input_path)
    docs = list(db.get_docs(nlp.vocab))
    msg.info(f"Found {len(docs)} docs in {input_path}")

    train_size, dev_size, test_size = split_size
    msg.info(f"Splitting docs using sizes: {split_size}")
    train, dev, test = _train_dev_test_split(
        docs, train_size, dev_size, test_size, shuffle, seed
    )
    datasets = {"train": train, "dev": dev, "test": test}

    msg.text(
        f"Done splitting the train ({len(train)}), dev ({len(dev)}), "
        f" and test ({len(test)}) datasets!"
    )

    for dataset, docs in datasets.items():
        output_path = output_dir / f"{input_path.stem}-{dataset}.spacy"
        db_new = DocBin(docs=docs)
        db_new.to_disk(output_path)
        msg.good(f"Saved {dataset} ({len(docs)}) dataset to {output_path}")


if __name__ == "__main__":
    typer.run(split_docs)
