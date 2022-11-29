from pathlib import Path
from typing import Callable, Iterable

import spacy
from spacy import registry, Vocab, Language
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.pipeline import EntityLinker

from wikid.scripts.kb import WikiKB


@spacy.registry.readers("EntityEnrichedCorpusReader.v1")
def create_docbin_reader(path: Path) -> Callable[[Language], Iterable[Example]]:
    """Returns Callable generating a corpus reader function that enriches read documents with the correct entities as
    specified in the corpus annotations.
    path (Path): Path to DocBin file with documents to prepare.
    """
    def read_docbin(nlp: Language) -> Iterable[Example]:
        """Read DocBin for training. Set all entities as they appear in the annotated corpus, but set entity type to
        NIL.
        nlp (Language): Pipeline to use for creating document used in EL from reference document.
        """
        nlp.disable_pipe("entity_linker")

        with nlp.select_pipes(disable="entity_linker"):
            for doc in DocBin().from_disk(path).get_docs(nlp.vocab):
                pred_doc = nlp(doc.text)
                pred_doc.ents = [
                    doc.char_span(ent.start_char, ent.end_char, label=EntityLinker.NIL, kb_id=EntityLinker.NIL)
                    for ent in doc.ents
                ]
                yield Example(pred_doc, doc)

        nlp.enable_pipe("entity_linker")

    return read_docbin


@registry.misc("spacy.WikiKBFromFile.v1")
def load_kb(kb_path: Path) -> Callable[[Vocab], WikiKB]:
    """Loads WikiKB instance from disk.
    kb_path (Path): Path to WikiKB path.
    RETURNS (Callable[[Vocab], WikiKB]): Callable generating WikiKB from disk.
    """
    def kb_from_file(_: Vocab) -> WikiKB:
        return WikiKB.generate_from_disk(path=kb_path)

    return kb_from_file


@registry.misc("spacy.EmptyWikiKB.v1")
def empty_wiki_kb() -> Callable[[Vocab, int], WikiKB]:
    """Generates empty WikiKB instance.
    RETURNS (Callable[[Vocab, int], WikiKB]): Callable generating WikiKB from disk.
    """
    def empty_kb_factory(vocab: Vocab, entity_vector_length: int):
        """Generates new WikiKB instance.
        Since WikiKB relies on an external DB file that we have no information on at this point, this instance will not
        have initialized its DB connection. Also, its parameters specified at init are arbitrarily chosen. This only
        serves to return a placeholder WikiKB instance to be overwritten using .from_bytes() or .from_disk().
        vocab (Vocab): Vocab instance.
        entity_vector_length (int): Entity vector length.
        """
        return WikiKB(
            vocab=vocab,
            entity_vector_length=entity_vector_length,
            db_path=Path("."),
            annoy_path=Path(".annoy"),
            language=".",
            establish_db_connection_at_init=False
        )

    return empty_kb_factory
