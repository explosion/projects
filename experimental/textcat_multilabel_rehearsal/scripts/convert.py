"""Convert textcat annotation from JSONL to spaCy v3 .spacy format."""
import srsly
import typer
import warnings
from pathlib import Path
from wasabi import msg

import spacy
from spacy.tokens import DocBin
from spacy import Language

REHEARSAL_LABELS = ["bread", "chicken", "food-safety", "storage-method"]
REMOVE_LABELS = [
    "OTHER"
]  # Removing the label OTHER because it has the most training data and is the most influential label


def convert_data(
    lang: str,
    input_train_path: Path,
    input_dev_path: Path,
    output_dir: Path,
):
    msg.info("Starting data conversion")
    msg.info(f"Splitting data with labels {REHEARSAL_LABELS} for rehearsal training.")
    msg.info(f"All models will use the same dev dataset")

    nlp = spacy.blank(lang)

    # Convert training data to .spacy format
    training_labels, training_labels_rehearsal, training_labels_all = convert(
        nlp, REHEARSAL_LABELS, input_train_path, output_dir, "train"
    )

    # Convert dev data to .spacy format
    dev_labels, dev_labels_rehearsal, dev_labels_all = convert(
        nlp, REHEARSAL_LABELS, input_dev_path, output_dir, "dev"
    )

    msg.good(f"{len(training_labels)} Labels found in dataset")
    data = [
        (
            label,
            training_labels[label],
            training_labels_rehearsal[label],
            dev_labels_all[label],
        )
        for label in training_labels
    ]
    header = (
        "Label",
        f"Training",
        f"Rehearsal",
        "Dev (100%)",
    )
    aligns = ("c", "c", "c")
    msg.table(data, header=header, divider=True, aligns=aligns)


def convert(
    nlp: Language,
    split_labels: list,
    input_path: Path,
    output_path: Path,
    output_prefix: str,
):
    labels = {}
    labels_rehearsal = {}
    labels_all = {}

    db = DocBin()
    db_rehearsal = DocBin()
    db_all = DocBin()

    for line in srsly.read_jsonl(input_path):
        doc = nlp.make_doc(line["text"])

        for del_label in REMOVE_LABELS:
            del line["cats"][del_label]

        doc.cats = line["cats"]
        db_all.add(doc)

        if not labels:
            for label in line["cats"]:
                labels[label] = 0
                labels_rehearsal[label] = 0
                labels_all[label] = 0

        for label in line["cats"]:
            if line["cats"][label] == 1.0:
                labels_all[label] += 1
                if label not in split_labels:
                    for doc_label in doc.cats:
                        if doc.cats[doc_label] == 1.0:
                            labels[doc_label] += 1
                    db.add(doc)
                    break
                else:
                    for doc_label in doc.cats:
                        if doc.cats[doc_label] == 1.0:
                            labels_rehearsal[doc_label] += 1
                    labels_rehearsal[label] += 1
                    db_rehearsal.add(doc)
                    break

    db_all.to_disk(output_path / (output_prefix + "_all.spacy"))
    db.to_disk(output_path / (output_prefix + ".spacy"))
    db_rehearsal.to_disk(output_path / (output_prefix + "_rehearse.spacy"))

    msg.good(f"Succesfully saved to {output_path}/{output_prefix}")

    return labels, labels_rehearsal, labels_all


if __name__ == "__main__":
    typer.run(convert_data)
