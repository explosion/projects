import functools
from pathlib import Path
from typing import Callable, Iterable

import spacy
from spacy import registry, Vocab, Language
from spacy.tokens import DocBin
from spacy.training import Example

from wikid.scripts.kb import WikiKB


@spacy.registry.readers("EntityEnrichedCorpusReader.v1")
def create_docbin_reader(path: Path, dataset_name: str) -> Callable[[Language], Iterable[Example]]:
    """Returns Callable generating a corpus reader function that enriches read documents with the correct entities as
    specified in the corpus annotations.
    path (Path): Path to DocBin file with documents to prepare.
    dataset_name (str): Dataset name/ID.
    """
    # todo read_files as local function?
    return functools.partial(read_files, path)


def read_files(path: Path, nlp: Language) -> Iterable[Example]:
    # todo docstring
    # we run the full pipeline and not just nlp.make_doc to ensure we have entities and sentences
    # which are needed during training of the entity linker.
    with nlp.select_pipes(disable="entity_linker"):
        doc_bin = DocBin().from_disk(path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        print(len(docs))
        for doc in docs:
            print("***", doc.ents, len(doc.ents))
            doc = nlp(doc.text)
            # todo set entities in predicted doc (with entity_id == NIL).
            if len(doc.ents):
                print(doc)
                for ent in doc.ents:
                    print("  ", ent.ent_id_, ent.start_char, ent.end_char)
                print("------")
            yield Example(nlp(doc.text), doc)


@registry.misc("spacy.WikiKBFromFile.v1")
def load_kb(kb_path: Path) -> Callable[[Vocab], WikiKB]:
    """Loads WikiKB instance from disk.
    kb_path (Path): Path to WikiKB path.
    RETURNS (Callable[[Vocab], WikiKB]): Callable generating WikiKB from disk.
    """
    def kb_from_file(_: Vocab) -> WikiKB:
        return WikiKB.generate_from_disk(path=kb_path)

    return kb_from_file
