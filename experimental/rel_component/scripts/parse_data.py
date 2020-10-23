import json

import typer
from spacy.lang.en import English
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer

msg = Printer()

SYMM_LABELS = ["Binds"]
MAP_LABELS = {
    "Pos-Reg": "Regulates",
    "Neg-Reg": "Regulates",
    "Reg": "Regulates",
    "No-rel": "Regulates",
    "Binds": "Binds",
}


def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})
    vocab = Vocab()

    train_docs = []
    dev_docs = []
    test_docs = []
    train_ids = set()
    dev_ids = set()
    test_ids = set()
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            span_starts = set()
            if example["answer"] == "accept":
                try:
                    # Parse the tokens
                    words = [t["text"] for t in example["tokens"]]
                    spaces = [t["ws"] for t in example["tokens"]]
                    doc = Doc(vocab, words=words, spaces=spaces)

                    # Parse the GGP entities
                    spans = example["spans"]
                    entities = []
                    for span in spans:
                        entity = doc.char_span(
                            span["start"], span["end"], label=span["label"]
                        )
                        entities.append(entity)
                        span_starts.add(span["token_start"])
                    doc.ents = entities

                    # Parse the relations
                    rels = {}
                    for x1 in span_starts:
                        for x2 in span_starts:
                            rels[(x1, x2)] = {}
                    relations = example["relations"]
                    for relation in relations:
                        start = relation["head_span"]["token_start"]
                        end = relation["child_span"]["token_start"]
                        label = relation["label"]
                        label = MAP_LABELS[label]
                        rels[(start, end)][label] = 1.0
                        if label in SYMM_LABELS:
                            rels[(end, start)][label] = 1.0

                    # The annotation is complete, so fill in zero's where the data is missing
                    for x1 in span_starts:
                        for x2 in span_starts:
                            for label in MAP_LABELS.values():
                                if rels[(x1, x2)].get(label, None) is None:
                                    rels[(x1, x2)][label] = 0.0
                    doc._.rel = rels

                    # use the original PMID/PMCID to decide on train/dev/test split
                    article_id = example["meta"]["source"]
                    article_id = article_id.replace("BioNLP 2011 Genia Shared Task, ", "")
                    article_id = article_id.replace(".txt", "")
                    article_id = article_id.split("-")[1]
                    if article_id.endswith("4"):
                        dev_ids.add(article_id)
                        dev_docs.append(doc)
                    elif article_id.endswith("3"):
                        test_ids.add(article_id)
                        test_docs.append(doc)
                    else:
                        train_ids.add(article_id)
                        train_docs.append(doc)
                except KeyError as e:
                    msg.fail(f"Skipping doc because of key error: {e} in {example['meta']['source']}")

    docbin = DocBin(docs=train_docs, store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"Wrote {len(train_docs)} training sentences from {len(train_ids)} articles to {train_file}"
    )

    docbin = DocBin(docs=dev_docs, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        f"Wrote {len(dev_docs)} dev sentences from {len(dev_ids)} articles to {dev_file}"
    )

    docbin = DocBin(docs=test_docs, store_user_data=True)
    docbin.to_disk(test_file)
    msg.info(
        f"Wrote {len(test_docs)} test sentences from {len(test_ids)} articles to {test_file}"
    )


if __name__ == "__main__":
    typer.run(main)
