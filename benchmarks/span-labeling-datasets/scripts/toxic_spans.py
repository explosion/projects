import ast
import csv
import random
from pathlib import Path
from typing import Optional

import spacy
import typer
from spacy.tokens import Doc, DocBin, SpanGroup
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option

ID = "toxic-spans"
TRAIN_SIZE = 0.8
DEV_SIZE = 0.1
TEST_SIZE = 0.1


def convert_toxic_spans(
    # fmt: off
    input_path: Path = Arg(..., help="Path to toxic_spans.csv", exists=True),
    output_dir: Path = Arg(..., help="Directory to save the train/dev/test files"),
    spans_key: str = Opt("sc", "--spans-key", help="Spans key to use when storing entities"),
    use_ents: bool = Opt(False, "--use-ents", "-e", help="Use Doc.ents, don't transfer to Doc.spans"),
    shuffle: bool = Opt(False, "--shuffle", "-sf", help="Shuffle the dataset before splitting"),
    seed: Optional[int] = Opt(None, "--seed", "-sd", help="Random seed for shuffling the data")
    # fmt: on
):
    """Convert the examples from the ToxicSpans dataset into the spaCy format

    For this experiment, we will be following the 80/10/10 train-dev-test split
    done in the paper: https://aclanthology.org/2022.acl-long.259/
    """
    with input_path.open(mode="r") as f:
        csv_reader = csv.DictReader(f)
        examples = []
        for row in csv_reader:
            examples.append(row)

    nlp = spacy.blank("en")
    docs = []
    for eg in examples:
        doc = nlp(eg["text_of_post"])
        span_indices = ast.literal_eval(eg["probability"])
        labels = ast.literal_eval(eg["type"])
        # In toxic-spans, we only have labels for spans where the annotator score
        # is > 0.5 (i.e., 2/3 annotators agree that a particular span is toxic)
        # That's why we filter it with this value
        span_indices = [idx for idx, p in span_indices.items() if p >= 0.5]
        if len(span_indices) > 0:
            spans = []
            for span_idx, label in zip(span_indices, labels):
                start, end = span_idx
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                spans.append(span)
            if use_ents:
                try:
                    doc.set_ents(spans)
                except ValueError:
                    # FIXME: There are 12 spans that overlap, but are mostly due to annotation errors
                    # Will find a way to include them (or remove them altogether)
                    pass
            else:
                group = SpanGroup(doc, name=spans_key, spans=spans)
                doc.spans[spans_key] = group
            docs.append(doc)

    msg.info(f"Processed {len(docs)} docs")

    # Split the dataset 80/10/10 based from the paper
    # TODO: Note that they actually did cross-validation here. For now
    # I'll do a straightforward split.
    if shuffle:
        msg.info("Shuffling docs before splitting")
        if seed:
            msg.info(f"Using random seed '{seed}'")
            random.seed(seed)
        random.shuffle(docs)

    # Separate training and test
    train_dev_size = int(len(docs) * (TRAIN_SIZE + DEV_SIZE))
    train_dev_docs = docs[:train_dev_size]
    test_docs = docs[train_dev_size:]

    # Get dev set from training
    train_size = int(len(docs) * TRAIN_SIZE)
    train_docs = train_dev_docs[:train_size]
    dev_docs = train_dev_docs[train_size:]

    msg.info(
        f"Split datasets into train ({len(train_docs)}), "
        f"dev ({len(dev_docs)}), and test ({len(test_docs)})."
    )

    datasets = [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]

    for name, _docs in datasets:
        doc_bin = DocBin(docs=_docs)
        output_file = output_dir / f"{ID}-{name}.spacy"
        doc_bin.to_disk(output_file)
        msg.good(f"Saved {name} dataset to {output_file}")


if __name__ == "__main__":
    typer.run(convert_toxic_spans)
