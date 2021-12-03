from multiprocessing import Value
import re
import random
import tarfile
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import spacy
import typer
from spacy.tokens import Doc, DocBin, Span
from spacy.util import get_words_and_spaces
from tqdm import tqdm
from wasabi import msg

from .constants import Directories as dirs


def _convert_to_doc(nlp, file_id: str) -> Doc:
    """Convert raw text and tokens into a Doc object

    Instead of using the text itself, we're using the tokens because
    there are some encoding errors in the '.text' file, rendering
    spacy.util.get_words_and_spaces to fail.
    """
    text = (dirs.TEXT_DIR / f"{file_id}.tokens").read_text()
    tokens = text.split(" ")

    # Create a Doc object
    words, spaces = get_words_and_spaces(words=tokens, text=text)
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    return doc


def _get_contiguous_tokens(labels: List[int]) -> List:
    """Get contiguous tokens that can be used for creating Spans

    The labelling scheme that the original dataset used is to set 1 if a
    specific token falls under a particular label, and 0 otherwise.  This
    function converts that logic to obtain the token range of these spans (i.e,
    get the range where there are continuous 1s). For example:

    [0, 1, 1, 0, 1, 1] -> [(1, 2), (4, 5)]

    Note that when making the actual spans, we still need to add +1 to the end
    token. This is because we want to pass the index of the first token after the
    span.
    """
    # [0, 1, 1, 0, 1, 1] -> [1, 2, 4, 5]
    indices = np.asarray([i for i, x in enumerate(labels) if x])
    if np.all((indices == 0)):
        return []
    else:
        # [1, 2, 4, 5] -> [(1, 2), (4, 5)]
        contig = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        # [(1, 2), 5] -> [(1, 2), (4, 5)]
        span_indices = [(c[0],) if len(c) == 1 else (c[0], c[-1]) for c in contig]
        return span_indices


def _read_annotations(f: Path) -> List[int]:
    annotations = f.read_text().split(",")
    return [int(a) for a in annotations]


def _attach_spans_to_doc(doc: Doc, file_id: str, span_key: str = "sc") -> Doc:
    """Attach the Spans to a given Doc based on the annotations

    doc (Doc): the spaCy Doc to attach the spans onto
    file_id (str): the file ID from where the spans will be sourced from.
    span_key (str): the span key to attach the spans onto, default is 'sc'.
    """
    directories = {
        "PARTICIPANTS": dirs.PARTICIPANTS_DIR,
        "INTERVENTIONS": dirs.INTERVENTIONS_DIR,
        "OUTCOMES": dirs.OUTCOMES_DIR,
    }
    annotations = {
        label: _read_annotations(list(dir_.glob(f"**/{file_id}_AGGREGATED.ann"))[0])
        for label, dir_ in directories.items()
    }

    # Sanity-check if all tokens are of the same length
    if not all([len(doc) == len(a) for a in annotations.values()]):
        msg.warn(f"Misaligned tokens in {file_id}")

    # Get the spans for each annotations file
    spans = []
    for annot, labels in annotations.items():
        indices = _get_contiguous_tokens(labels)
        if indices:
            for idx in indices:
                start = idx[0]
                end = idx[0] if len(idx) == 1 else idx[-1]
                # We add +1 because we want the index AFTER the span
                spans.append(Span(doc, start, end + 1, label=annot))
    doc.spans[span_key] = spans
    return doc


def _get_ids(s: str) -> str:
    """Remove the suffix from the files to obtain the ids"""
    return re.sub("\_AGGREGATED", "", s)


def to_spacy(file_id, nlp) -> Doc:
    """Convert a file id into a spaCy doc. A convenience function"""
    doc = _convert_to_doc(nlp, file_id)
    doc = _attach_spans_to_doc(doc, file_id)
    return doc


def main(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to the tar.gz file."
    ),
    output_path: Path = typer.Argument(
        ..., dir_okay=True, help="Output directory to save the spaCy serialized format."
    ),
    train_size: float = typer.Option(
        0.8, show_default=True, help="Ratio of training data"
    ),
    shuffle: bool = typer.Option(
        True,
        show_default=True,
        help="Shuffle training data before splitting it to dev.",
    ),
    random_seed: int = typer.Option(
        42, show_default=True, help="Random seed to control shuffling."
    ),
):
    """Convert the downloaded EBM-NLP data into the spaCy binary format"""
    # Extract the tar.gz file
    msg.text(f"Extracting contents from zipped file: {input_path}")
    with tarfile.open(input_path) as f:
        for member in tqdm(f.getmembers(), total=len(f.getmembers())):
            f.extract(member, path=dirs.ASSETS_DIR)

    # Read each file and convert them into a spaCy doc
    train_file_ids = [
        _get_ids(file.stem)
        for file in (dirs.PARTICIPANTS_DIR / "train").glob("*_AGGREGATED.ann")
    ]
    test_file_ids = [
        _get_ids(file.stem)
        for file in (dirs.PARTICIPANTS_DIR / "test" / "gold").glob("*_AGGREGATED.ann")
    ]
    msg.text(f"Found {len(train_file_ids)} train and {len(test_file_ids)} test files")

    # Shuffle and split training data into train and dev
    random.seed(random_seed)
    if shuffle:
        msg.text(f"Shuffling the training IDs before splitting (seed={random_seed})")
        random.shuffle(train_file_ids)
    training_count = int(len(train_file_ids) * train_size)
    dev_file_ids = train_file_ids[training_count:]
    train_file_ids = train_file_ids[:training_count]
    msg.text(
        f"Split files into {len(train_file_ids)} train and {len(dev_file_ids)} dev"
    )

    # Convert to DocBin then save to disk
    nlp = spacy.blank("en")

    train_doc_bin = DocBin()
    for train_id in tqdm(train_file_ids, desc="Parsing train files"):
        try:
            doc = to_spacy(train_id, nlp)
        except ValueError as e:
            msg.fail(f"Error in {train_id}: {e}")
        else:
            train_doc_bin.add(doc)
    train_doc_bin.to_disk(output_path / "train.spacy")
    msg.good(f"Saved train docs to corpus")

    dev_doc_bin = DocBin()
    for dev_id in tqdm(dev_file_ids, desc="Parsing dev files"):
        try:
            doc = to_spacy(dev_id, nlp)
        except ValueError as e:
            msg.fail(f"Error in {dev_id}: {e}")
        else:
            dev_doc_bin.add(doc)
    dev_doc_bin.to_disk(output_path / "dev.spacy")
    msg.good(f"Saved dev docs to corpus")

    test_doc_bin = DocBin()
    for test_id in tqdm(test_file_ids, desc="Parsing test files"):
        try:
            doc = to_spacy(test_id, nlp)
        except ValueError as e:
            msg.fail(f"Error in {test_id}: {e}")
        else:
            test_doc_bin.add(doc)
    test_doc_bin.to_disk(output_path / "test.spacy")
    msg.good(f"Saved test docs to corpus")


if __name__ == "__main__":
    typer.run(main)
