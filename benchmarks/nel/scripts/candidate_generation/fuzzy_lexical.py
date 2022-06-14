""" Candidate generation via distance in lexical space. """
from itertools import chain
from typing import Iterable

from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span
from .base import NearestNeighborCandidateSelector
from fuzzyset import FuzzySet


class CandidateSelector(NearestNeighborCandidateSelector):
    """ Callable object selecting candidates as nearest neighbours in lexical space. """

    def _init_container(self, dataset_id: str, kb: KnowledgeBase, max_n_candidates: int, **kwargs) -> FuzzySet:
        return FuzzySet(kb.get_alias_strings())

    def _fetch_candidates(
        self, dataset_id: str, span: Span, kb: KnowledgeBase, max_n_candidates: int, **kwargs
    ) -> Iterable[int]:
        return chain.from_iterable([
            kb.get_alias_candidates(entry[1])
            for entry in self._container[dataset_id].get(span.text, [])[:max_n_candidates]
        ])
