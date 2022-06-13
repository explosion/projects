""" Custom functions to be hooked up into the registry. """
from functools import partial
from typing import Iterable, Callable
import typing
import spacy
from spacy.kb import Candidate, KnowledgeBase
from spacy.tokens import Span

# More elegant way to resolve import conflicts between training and evaluation calls?
try:
    import scripts.candidate_generation.embeddings as embedding_candidate_generation
    import scripts.candidate_generation.fuzzy_lexical as fuzzy_lexical_candidate_generation
except ModuleNotFoundError:
    import candidate_generation.embeddings as embedding_candidate_generation
    import candidate_generation.fuzzy_lexical as fuzzy_lexical_candidate_generation

embedding_candidate_selector = embedding_candidate_generation.CandidateSelector()
fuzzy_lexical_candidate_selector = fuzzy_lexical_candidate_generation.CandidateSelector()


@spacy.registry.misc("EmbeddingGetCandidates.v1")
def create_candidates_via_embeddings(dataset_id: str, k: int) -> Callable[[KnowledgeBase, Span], Iterable[Candidate]]:
    """ Returns Callable for identification of candidates via NN search in embedding space.
    dataset_id (str): Dataset ID.
    k (int): Numbers of nearest neighbours to query.
    RETURNS (Callable[[KnowledgeBase, Span], Iterable[Candidate]]): Callable for identification of entity candidates.
    """

    # More elegant way to enforce proper typing for partial object?
    return typing.cast(
        Callable[[KnowledgeBase, Span], Iterable[Candidate]],
        partial(embedding_candidate_selector, dataset_id, k)
    )


@spacy.registry.misc("FuzzyStringGetCandidates.v1")
def create_candidates_via_embeddings(dataset_id: str, k: int) -> Callable[[KnowledgeBase, Span], Iterable[Candidate]]:
    """ Returns Callable for identification of candidates via NN search in lexical space.
    dataset_id (str): Dataset ID.
    k (int): Numbers of nearest neighbours to query.
    RETURNS (Callable[[KnowledgeBase, Span], Iterable[Candidate]]): Callable for identification of entity candidates.
    """

    # More elegant way to enforce proper typing for partial object?
    return typing.cast(
        Callable[[KnowledgeBase, Span], Iterable[Candidate]],
        partial(fuzzy_lexical_candidate_selector, dataset_id, k)
    )
