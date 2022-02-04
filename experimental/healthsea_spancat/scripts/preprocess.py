from spacy.tokens import DocBin, Span
import spacy

from wasabi import Printer
from wasabi import table

import json
from pathlib import Path
import typer

msg = Printer()

def main(
    json_loc: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
):
    """Parse the annotations into a training and development set for NER and Spancat."""

    empty_docs = []
    docs = []
    nlp = spacy.blank("en")
    total_span_count = {}
    max_span_length = 0

    # Load dataset
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)

            if example["answer"] == "accept": 
                doc = nlp(example["text"])
                spans = []

                if "spans" in example:
                    for span in example["spans"]:
                        spans.append(Span(doc, span["token_start"], span["token_end"]+1, span["label"]))
                        
                        if span["label"] not in total_span_count:
                            total_span_count[span["label"]] = 0

                        total_span_count[span["label"]] += 1
                        
                        span_length = (span["token_end"]+1)-span["token_start"]
                        if span_length > max_span_length:
                            max_span_length = span_length

                doc.set_ents(spans)
                doc.spans["health_aspects"] = spans

                if len(doc.ents) > 0:
                    docs.append(doc)
                else:
                    empty_docs.append(doc)

    # Split
    train = []
    dev = []
    table_data = []

    split = int(len(docs) * eval_split)
    train = docs[split:] + empty_docs[split:]
    dev = docs[:split] + empty_docs[:split]

    # Printing
    msg.divider("Dataset summary")

    msg.info(f"Docs with spans: {len(docs)}")
    msg.info(f"Docs without spans: {len(empty_docs)}")

    for span_label in total_span_count:
        msg.info(f"Total span count [{span_label}]: {total_span_count[span_label]}")

    msg.info(f"Max span length: {max_span_length}")
    msg.info(f"Evaluation split: {eval_split}")
    table_data.append((len(docs)+len(empty_docs), len(train), len(dev)))
    header = ("Total", "Training", "Development")
    print(table(table_data, header=header, divider=True))

    # Save to disk
    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.good(f"Parsing complete")


if __name__ == "__main__":
    typer.run(main)
