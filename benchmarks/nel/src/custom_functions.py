from pathlib import Path
from typing import Callable, Iterable

import spacy
from spacy import registry, Vocab, Language
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.pipeline import EntityLinker

from wikid.src.kb import WikiKB


@spacy.registry.readers("EntityEnrichedCorpusReader.v1")
def create_docbin_reader(path: Path, path_nlp_base: Path) -> Callable[[Language], Iterable[Example]]:
    """Returns Callable generating a corpus reader function that enriches read documents with the correct entities as
    specified in the corpus annotations.
    path (Path): Path to DocBin file with documents to prepare.
    path_nlp_base (Path): Path to pipeline for tokenization/sentence.
    """
    def read_docbin(_: Language) -> Iterable[Example]:
        """Read DocBin for training. Set all entities as they appear in the annotated corpus, but set entity type to
        NIL.
        nlp (Language): Pipeline to use for creating document used in EL from reference document.
        """
        nlp = spacy.load(path_nlp_base, enable=["senter"], config={"nlp.disabled": []})

        for doc in DocBin().from_disk(path).get_docs(nlp.vocab):
            pred_doc = nlp(doc.text)
            pred_doc.ents = [
                pred_doc.char_span(ent.start_char, ent.end_char, label=EntityLinker.NIL, kb_id=EntityLinker.NIL)
                for ent in doc.ents
            ]
            sents = list(pred_doc.sents)
            sents_orig = list(doc.sents)
            assert len(sents) == len(sents_orig)
            assert len(sents) > 0 and len(sents_orig) > 0
            assert all([ent is not None for ent in pred_doc.ents])
            assert len(doc.ents) == len(pred_doc.ents)
            assert len(doc.ents) > 0

            yield Example(pred_doc, doc)

    return read_docbin


@spacy.registry.readers("EntityEnrichedCorpusReader.v2")
def create_docbin_reader(path: Path, path_nlp_base: Path) -> Callable[[Language], Iterable[Example]]:
    """Returns Callable generating a corpus reader function that enriches read documents with the correct entities as
    specified in the corpus annotations.
    path (Path): Path to DocBin file with documents to prepare.
    path_nlp_base (Path): Path to pipeline for tokenization/sentence.
    """
    def read_docbin(_: Language) -> Iterable[Example]:
        """Read DocBin for training. Set all entities as they appear in the annotated corpus, but set entity type and KB
        ID to NIL.
        nlp (Language): Pipeline to use for creating document used in EL from reference document.
        """
        nlp = spacy.load(path_nlp_base)

        for example in spacy.training.Corpus(path)(nlp):
            example.predicted = nlp(example.predicted)
            example.predicted.ents = [
                example.predicted.char_span(ent.start_char, ent.end_char, label=EntityLinker.NIL, kb_id=EntityLinker.NIL)
                for ent in example.reference.ents
            ]
            sents = list(example.predicted.sents)
            sents_orig = list(example.reference.sents)

            # if len(sents) != len(sents_orig):
            #     for i in range(max(len(sents), len(sents_orig))):
            #         if i < len(sents):
            #             print(sents[i])
            #         else:
            #             print("out")
            #         if i < len(sents_orig):
            #             print(sents_orig[i])
            #         else:
            #             print("out")
            #         print("-----")
            #     x = 3

            assert len(sents) == len(sents_orig)
            assert len(sents) > 0 and len(sents_orig) > 0
            assert all([ent is not None for ent in example.predicted.ents])
            assert len(example.reference.ents) == len(example.predicted.ents)
            assert len(example.reference.ents) > 0

            yield Example(example.predicted, example.reference)

    return read_docbin


@registry.misc("spacy.WikiKBFromFile.v1")
def load_kb(kb_path: Path) -> Callable[[Vocab], WikiKB]:
    """Loads WikiKB instance from disk.
    kb_path (Path): Path to WikiKB path.
    mention_candidates_path (Path): Path to pre-computed file with candidates per mention.
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
