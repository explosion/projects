import typer
import spacy
import os

from typing import Set, Sequence, Optional

from tqdm import tqdm
from wasabi import msg
from spacy.tokens import DocBin, Doc
from _util import info


def _mark_as_missing(
    docs: Sequence[Doc], seen: Set[str], mark_seen: bool = True, *, total: Optional[int] = None
) -> DocBin:
    """
    Marks some of the ents in the input Docs as missing.
    If 'mark_seen' is True then it marks entities in
    'seen' as missing otherwise it marks the entities not in 'seen'
    as missing.
    """
    new_docbin = DocBin()
    for doc in tqdm(docs, total=total):
        if len(doc.ents) != 0:
            missing = []
            for ent in doc.ents:
                if not mark_seen ^ (ent.text in seen):
                    missing.append(ent)
            doc.set_ents([], missing=missing, default="unmodified")
        new_docbin.add(doc)
    return new_docbin


def split_seen_unseen():
    datasets = info("ner")
    for _, dataset in datasets.items():
        trainbin, devbin, testbin = dataset.load()
        msg.good(f"Loaded data set {dataset.source}.")
        nlp = spacy.blank(dataset.lang)
        train_entities = set()
        all_ents = 0
        for doc in tqdm(trainbin.get_docs(nlp.vocab), total=len(trainbin)):
            all_ents += len(doc.ents)
            entities = {span.text for span in doc.ents}
            train_entities.update(entities)
        msg.good(
            f"Collected {len(train_entities)} unique "
            f"entities from a total of {all_ents}."
        )
        unseen_dev = _mark_as_missing(
            docs=devbin.get_docs(nlp.vocab),
            seen=train_entities,
            mark_seen=True,
            total=len(devbin),
        )
        seen_dev = _mark_as_missing(
            docs=devbin.get_docs(nlp.vocab),
            seen=train_entities,
            mark_seen=False,
            total=len(devbin),
        )
        unseen_test = _mark_as_missing(
            docs=testbin.get_docs(nlp.vocab),
            seen=train_entities,
            mark_seen=True,
            total=len(testbin),
        )
        seen_test = _mark_as_missing(
            docs=testbin.get_docs(nlp.vocab),
            seen=train_entities,
            mark_seen=False,
            total=len(testbin),
        )
        unseen_dev_path = os.path.join("unseen", f"{dataset.source}-dev-unseen.spacy")
        unseen_test_path = os.path.join("unseen", f"{dataset.source}-test-unseen.spacy")
        seen_dev_path = os.path.join("unseen", f"{dataset.source}-dev-seen.spacy")
        seen_test_path = os.path.join("unseen", f"{dataset.source}-test-seen.spacy")
        seen_dev.to_disk(seen_dev_path)
        seen_test.to_disk(seen_test_path)
        unseen_dev.to_disk(unseen_dev_path)
        unseen_test.to_disk(unseen_test_path)


if __name__ == "__main__":
    typer.run(split_seen_unseen)
