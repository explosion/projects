from spacy.tokens import DocBin, Span
from spacy.lang.en import English

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
    """Parse the textcat annotations into a training and development set."""

    docs = []
    nlp = English()

    # Load dataset
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)

            if example["answer"] == "accept": 
                doc = nlp(example["text"])
                
                if "spans" in example:
                    for span in example["spans"]:
                        doc.set_ents([Span(doc, span["token_start"], span["token_end"]+1, span["label"])])

                doc.spans["health_aspects"] = list(doc.ents)
                docs.append(doc)

    # Split
    train = []
    dev = []
    table_data = []

    split = int(len(docs) * eval_split)
    train = docs[split:]
    dev = docs[:split]

    # Printing
    msg.divider("NER dataset summary")
    msg.info(f"Evaluation split: {eval_split}")
    table_data.append((len(docs), len(train), len(dev)))
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
