""" Custom functions to be hooked up into the registry. """
from functools import partial

import spacy
from typing import Iterable, Callable

import typing
from spacy.kb import Candidate, KnowledgeBase
from spacy.tokens import Span
from scripts.candidate_generation.embeddings import create_candidates


@spacy.registry.misc("EmbeddingGetCandidates.v1")
def create_candidates_via_embeddings(dataset_name: str) -> Callable[[KnowledgeBase, Span], Iterable[Candidate]]:
    """ Returns Callable for identification of candidates via their embeddings.
    dataset_name (str): Dataset name.
    RETURNS (Callable[[KnowledgeBase, Span], Iterable[Candidate]]): Callable for identification of entity candidates.
    """

    # More elegant way to enforce proper typing for partial object?
    return typing.cast(
        Callable[[KnowledgeBase, Span], Iterable[Candidate]],
        partial(create_candidates, dataset_name)
    )
