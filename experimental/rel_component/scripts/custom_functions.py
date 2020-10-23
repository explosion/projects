from functools import partial
from pathlib import Path
from typing import Iterable, Callable
from spacy import registry
from spacy.training import Example
from spacy.tokens import DocBin

from scripts.rel_pipe import make_relation_extractor  # make the factory work
from scripts.rel_model import create_relation_model, create_candidates, create_layer  # make the config work


@registry.readers("Gold_ents_Corpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    # we add the gold GGP annotations to the "predictions" doc as we do not attempt to predict these
    doc_bin = DocBin().from_disk(file)
    docs = doc_bin.get_docs(nlp.vocab)
    for gold in docs:
        pred = nlp.make_doc(gold.text)
        pred.ents = gold.ents
        yield Example(pred, gold)
