from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import DocBin, Doc

# make the factory work
from scripts.rel_pipe import make_relation_extractor

# make the config work
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


@spacy.registry.readers("Gold_ents_Corpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    """Custom reader that keeps the tokenization of the gold data,
    and also adds the gold GGP annotations as we do not attempt to predict these."""
    doc_bin = DocBin().from_disk(file)
    docs = doc_bin.get_docs(nlp.vocab)
    for gold in docs:
        pred = Doc(
            nlp.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
        )
        pred.ents = gold.ents
        yield Example(pred, gold)
