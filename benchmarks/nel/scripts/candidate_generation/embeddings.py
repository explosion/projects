""" Candidate generation. """

from typing import Iterator
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span

from scripts.datasets.dataset import Dataset


def create_candidates(dataset_id: str, kb: KnowledgeBase, span: Span) -> Iterator[Candidate]:
    """ Identifies entity candidates via their embeddings.
    kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
    span (Span): Span to whom match potential entity candidates to.
    """

    # todo @RM
    #  - load (and cache) NLP
    #  - add temp getters to KB class to access all information necessary to create candidate objects
    #  - infer vector for span.text
    #  - compute distances between KB entity vectors and span.text vector
    #  - define offcut/top-n in config, return relevant candidates
    #  - evaluate results, compare with get_candidates()
    nlp_path = Dataset.assemble_paths(dataset_id)["nlp_best"]

    return kb.get_alias_candidates(span.text)

