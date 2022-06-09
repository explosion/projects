""" Custom functions to be hooked up into the registry. """
from functools import partial

import spacy
from typing import Iterable, Callable

import typing
from spacy.kb import Candidate, KnowledgeBase
from spacy.tokens import Span
import scripts.candidate_generation.embeddings

embedding_candidate_selector = scripts.candidate_generation.embeddings.CandidateSelector()


@spacy.registry.misc("EmbeddingGetCandidates.v1")
def create_candidates_via_embeddings(dataset_id: str, k: int) -> Callable[[KnowledgeBase, Span], Iterable[Candidate]]:
    """ Returns Callable for identification of candidates via their embeddings.
    dataset_id (str): Dataset ID.
    k (int): Numbers of nearest neighbours to query.
    RETURNS (Callable[[KnowledgeBase, Span], Iterable[Candidate]]): Callable for identification of entity candidates.
    """

    # More elegant way to enforce proper typing for partial object?
    return typing.cast(
        Callable[[KnowledgeBase, Span], Iterable[Candidate]],
        partial(embedding_candidate_selector, dataset_id, k)
    )
