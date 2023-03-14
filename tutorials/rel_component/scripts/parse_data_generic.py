# This script was derived from parse_data.py but made more generic as a template for various REL parsing needs

import json
import random
import typer
from pathlib import Path

from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer

msg = Printer()

# TODO: define your labels used for annotation either as "symmetrical" or "directed"
SYMM_LABELS = ["Binds"]
DIRECTED_LABELS = ["Regulates", "Impacts"]

# TODO: define splits for train/dev/test. What is not in test or dev, will be used as train.
test_portion = 0.2
dev_portion = 0.3

# TODO: set this bool to False if you didn't annotate all relations in all sentences.
# If it's true, entities that were not annotated as related will be used as negative examples.
is_complete = True


def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})
    vocab = Vocab()

    docs = {"train": [], "dev": [], "test": []}
    count_all = {"train": 0, "dev": 0, "test": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0}

    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            span_starts = set()
            if example["answer"] == "accept":
                neg = 0
                pos = 0
                # Parse the tokens
                words = [t["text"] for t in example["tokens"]]
                spaces = [t["ws"] for t in example["tokens"]]
                doc = Doc(vocab, words=words, spaces=spaces)

                # Parse the entities
                spans = example["spans"]
                entities = []
                span_end_to_start = {}
                for span in spans:
                    entity = doc.char_span(
                        span["start"], span["end"], label=span["label"]
                    )
                    span_end_to_start[span["token_end"]] = span["token_start"]
                    entities.append(entity)
                    span_starts.add(span["token_start"])
                if not entities:
                    msg.warn("Could not parse any entities from the JSON file.")
                doc.ents = entities

                # Parse the relations
                rels = {}
                for x1 in span_starts:
                    for x2 in span_starts:
                        rels[(x1, x2)] = {}
                relations = example["relations"]
                for relation in relations:
                    # Ignoring relations that are not between spans (they are annotated on the token level
                    if not relation["head"] in span_end_to_start or not relation["child"] in span_end_to_start:
                        msg.warn(f"This script only supports relationships between annotated entities.")
                        break
                    # the 'head' and 'child' annotations refer to the end token in the span
                    # but we want the first token
                    start = span_end_to_start[relation["head"]]
                    end = span_end_to_start[relation["child"]]
                    label = relation["label"]
                    if label not in SYMM_LABELS + DIRECTED_LABELS:
                        msg.warn(f"Found label '{label}' not defined in SYMM_LABELS or DIRECTED_LABELS - skipping")
                        break
                    if label not in rels[(start, end)]:
                        rels[(start, end)][label] = 1.0
                        pos += 1
                    if label in SYMM_LABELS:
                        if label not in rels[(end, start)]:
                            rels[(end, start)][label] = 1.0
                            pos += 1

                # If the annotation is complete, fill in zero's where the data is missing
                if is_complete:
                    for x1 in span_starts:
                        for x2 in span_starts:
                            for label in SYMM_LABELS + DIRECTED_LABELS:
                                if label not in rels[(x1, x2)]:
                                    neg += 1
                                    rels[(x1, x2)][label] = 0.0
                doc._.rel = rels

                # only keeping documents with at least 1 positive case
                if pos > 0:
                    # create the train/dev/test split randomly
                    # Note that this is not good practice as instances from the same article
                    # may end up in different splits. Ideally, change this method to keep
                    # documents together in one split (as in the original parse_data.py)
                    if random.random() < test_portion:
                        docs["test"].append(doc)
                        count_pos["test"] += pos
                        count_all["test"] += pos + neg
                    elif random.random() < (test_portion + dev_portion):
                        docs["dev"].append(doc)
                        count_pos["dev"] += pos
                        count_all["dev"] += pos + neg
                    else:
                        docs["train"].append(doc)
                        count_pos["train"] += pos
                        count_all["train"] += pos + neg

    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"{len(docs['train'])} training sentences, "
        f"{count_pos['train']}/{count_all['train']} pos instances."
    )

    docbin = DocBin(docs=docs["dev"], store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        f"{len(docs['dev'])} dev sentences, "
        f"{count_pos['dev']}/{count_all['dev']} pos instances."
    )

    docbin = DocBin(docs=docs["test"], store_user_data=True)
    docbin.to_disk(test_file)
    msg.info(
        f"{len(docs['test'])} test sentences, "
        f"{count_pos['test']}/{count_all['test']} pos instances."
    )


if __name__ == "__main__":
    typer.run(main)
