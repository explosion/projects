""" Custom functions to be hooked up into the registry. """
from functools import partial

from typing import Iterable, Callable
import typing
import spacy
from spacy.kb import Candidate, KnowledgeBase
from spacy.tokens import Span

from scripts.candidate_generation import embeddings
from scripts.candidate_generation import lexical

embedding_candidate_selector = embeddings.EmbeddingCandidateSelector()
fuzzy_lexical_candidate_selector = lexical.LexicalCandidateSelector()


@spacy.registry.misc("EmbeddingGetCandidates.v1")
def create_candidates_via_embeddings(
    dataset_name: str, max_n_candidates: int, lexical_similarity_cutoff: float
) -> Callable[[KnowledgeBase, Span], Iterable[Candidate]]:
    """Returns Callable for identification of candidates via their embeddings.
    dataset_name (str): Dataset name.
    max_n_candidates (int): Numbers of nearest neighbours to query.
    RETURNS (Callable[[KnowledgeBase, Span], Iterable[Candidate]]): Callable for identification of entity candidates.
    """

    # More elegant way to enforce proper typing for partial object?
    return typing.cast(
        Callable[[KnowledgeBase, Span], Iterable[Candidate]],
        partial(
            embedding_candidate_selector,
            dataset_id=dataset_name,
            max_n_candidates=max_n_candidates,
            lexical_similarity_cutoff=lexical_similarity_cutoff,
        ),
    )


@spacy.registry.misc("FuzzyStringGetCandidates.v1")
def create_candidates_via_fuzzy_string_matching(
    dataset_name: str, max_n_candidates: int, similarity_cutoff: float
) -> Callable[[KnowledgeBase, Span], Iterable[Candidate]]:
    """Returns Callable for identification of candidates via NN search in lexical space.
    dataset_name (str): Dataset name.
    max_n_candidates (int): Numbers of nearest neighbours to query.
    similarity_cutoff (float): Similarity value below which candidates won't be included.
    RETURNS (Callable[[KnowledgeBase, Span], Iterable[Candidate]]): Callable for identification of entity candidates.
    """

    assert 0 <= similarity_cutoff <= 1

    # More elegant way to enforce proper typing for partial object?
    return typing.cast(
        Callable[[KnowledgeBase, Span], Iterable[Candidate]],
        partial(
            fuzzy_lexical_candidate_selector,
            dataset_id=dataset_name,
            max_n_candidates=max_n_candidates,
            similarity_cutoff=similarity_cutoff,
        ),
    )
