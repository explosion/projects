import random
import re
import tarfile
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
from .constants import Pipeline
from .constants import SPAN_KEY


DIRECTORIES = {
    "PARTICIPANTS": dirs.PARTICIPANTS_DIR,
    "INTERVENTIONS": dirs.INTERVENTIONS_DIR,
    "OUTCOMES": dirs.OUTCOMES_DIR,
}


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


def _create_spans_from_annotations(
    doc: Doc, file_id: str, entities: List[str] = DIRECTORIES.keys()
) -> List[Span]:
    """Read the annotations file and create spaCy spans out of that

    doc (Doc): the spaCy Doc to attach the spans onto.
    file_id (str): the file ID from where the spans will be sourced from.
    entities (List[str]): filter the spans based on what kind of entities you just need.
    """
    # filter entities based on what you just want to get
    directories = {k: v for k, v in DIRECTORIES.items() if k in entities}
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

    return spans


def _get_ids(s: str) -> str:
    """Remove the suffix from the files to obtain the ids"""
    return re.sub("\_AGGREGATED", "", s)


def create_docbin(
    nlp, file_ids: List[str], dataset_type: str, output_path: Path, pipeline: Pipeline
):
    """Convert to DocBin and save them to disk"""
    if pipeline == Pipeline.spancat:
        doc_bin = DocBin()
        for id_ in tqdm(file_ids, desc=f"Parsing {dataset_type} files"):
            try:
                doc = _convert_to_doc(nlp, id_)
                spans = _create_spans_from_annotations(doc, id_)
                doc.spans[SPAN_KEY] = spans
            except ValueError as e:
                msg.fail(f"Error in {id_}: {e}")
            else:
                doc_bin.add(doc)

        fp = output_path / f"{pipeline}_{dataset_type}.spacy"
        doc_bin.to_disk(fp)
        msg.good(f"Saved {dataset_type} Docs to corpus")
    elif pipeline == Pipeline.ner:
        entities = DIRECTORIES.keys()
        for entity in entities:
            msg.text(f"Creating Docs for {entity}")
            doc_bin = DocBin()
            doc_entities_count = 0
            for id_ in tqdm(
                file_ids, desc=f"Parsing {dataset_type} files for {entity}"
            ):
                try:
                    doc = _convert_to_doc(nlp, id_)
                    spans = _create_spans_from_annotations(doc, id_, entities=[entity])
                    doc.ents = spans
                    doc_entities_count += len(spans)
                except ValueError as e:
                    msg.fail(f"Error in {id_}: {e}")
                else:
                    doc_bin.add(doc)
            fp = output_path / f"{pipeline}_{dataset_type}_{entity[0].lower()}.spacy"
            doc_bin.to_disk(fp)
            msg.good(
                f"Saved {dataset_type} [{doc_entities_count} {entity}] Docs to corpus"
            )

    elif pipeline == Pipeline.combined:
        raise NotImplementedError
    else:
        msg.fail(f"Unknown pipeline type: {pipeline}")


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
    pipeline: Pipeline = typer.Option(
        Pipeline.spancat,
        show_default=True,
        help="Pipeline you will create. The created Docs will depend on this value",
    ),
):
    """Convert the downloaded EBM-NLP data into the spaCy binary format"""
    # Extract the tar.gz file
    msg.text(f"Extracting contents from zipped file: {input_path}")
    with tarfile.open(input_path) as f:
        for member in tqdm(f.getmembers(), total=len(f.getmembers())):
            f.extract(member, path=dirs.ASSETS_DIR)

    # Read each file and get the IDs
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
    create_docbin(nlp, train_file_ids, "train", output_path, pipeline)
    create_docbin(nlp, dev_file_ids, "dev", output_path, pipeline)
    create_docbin(nlp, test_file_ids, "test", output_path, pipeline)


if __name__ == "__main__":
    typer.run(main)
