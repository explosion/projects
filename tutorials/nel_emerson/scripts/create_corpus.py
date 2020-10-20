import typer
import json
from collections import Counter
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Span


def main(json_loc: Path, nlp_dir: Path, train_corpus: Path, test_corpus: Path):
    """ Step 2: Once we have done the manual annotations with Prodigy, create corpora in spaCy format. """
    nlp = spacy.load(nlp_dir, exclude="parser, tagger")
    docs = []
    gold_ids = []
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            sentence = example["text"]
            if example["answer"] == "accept":
                QID = example["accept"][0]
                doc = nlp.make_doc(sentence)
                gold_ids.append(QID)
                # we assume only 1 annotated span per sentence, and only 1 KB ID per span
                entity = doc.char_span(
                    example["spans"][0]["start"],
                    example["spans"][0]["end"],
                    label=example["spans"][0]["label"],
                    kb_id=QID,
                )
                doc.ents = [entity]
                for i, t in enumerate(doc):
                    doc[i].is_sent_start = i == 0
                docs.append(doc)

    print("Statistics of manually annotated data:")
    print(Counter(gold_ids))
    print()

    train_docs = DocBin()
    test_docs = DocBin()
    for QID in ["Q312545", "Q48226", "Q215952"]:
        indices = [i for i, j in enumerate(gold_ids) if j == QID]
        # first 8 in training
        for index in indices[0:8]:
            train_docs.add(docs[index])
        # last 2 in test
        for index in indices[8:10]:
            test_docs.add(docs[index])

    train_docs.to_disk(train_corpus)
    test_docs.to_disk(test_corpus)


if __name__ == "__main__":
    typer.run(main)
