""" Candidate generation via distance in lexical space. """
from typing import Iterable

from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span
from .base import NearestNeighborCandidateSelector
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
        # todo 3. get rid of entity pickle files (move loading, stats to compile_corpora)
        # todo 4. re-evaluate efficacy of fuzzy string lookup (memory? access time?)
        # todo also: push forward spacy NEL changes - add mechanism for pushing back entity sets instead of single
        #  entities - how?

        hits = self._lookup_struct.get(span.text, [])
        all_cands = [
            kb.get_alias_candidates(entry[1]) for entry in self._lookup_struct.get(span.text, [])
            if entry[0] >= similarity_cutoff
        ][:max_n_candidates]
        x = 3
        return {cand for cands_for_alias in all_cands for cand in cands_for_alias}
