""" Candidate generation via distance in lexical space. """
from typing import Iterable

from spacy.tokens import Span
from .base import NearestNeighborCandidateSelector
from ..compat import KnowledgeBase
from cfuzzyset import cFuzzySet as FuzzySet


class LexicalCandidateSelector(NearestNeighborCandidateSelector):
    """Callable object selecting candidates as nearest neighbours in lexical space."""

    def _init_lookup_structure(self, kb: KnowledgeBase, max_n_candidates: int, **kwargs) -> FuzzySet:
        return FuzzySet(kb.get_alias_strings())

    def _fetch_candidates(
        self,
        dataset_id: str,
        span: Span,
        kb: KnowledgeBase,
        max_n_candidates: int,
        similarity_cutoff: float = 0.5,
    ) -> Iterable[int]:
        all_cands = [
            kb.get_alias_candidates(entry[1]) for entry in self._lookup_struct.get(span.text, [])
            if entry[0] >= similarity_cutoff
        ][:max_n_candidates]

        return {cand for cands_for_alias in all_cands for cand in cands_for_alias}
