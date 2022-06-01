from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import DocBin


@spacy.registry.readers("MyCorpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    # we run the full pipeline and not just nlp.make_doc to ensure we have entities and sentences
    # which are needed during training of the entity linker
    with nlp.select_pipes(disable="entity_linker"):
        doc_bin = DocBin().from_disk(file)
        docs = doc_bin.get_docs(nlp.vocab)
        for doc in docs:
            yield Example(nlp(doc.text), doc)
