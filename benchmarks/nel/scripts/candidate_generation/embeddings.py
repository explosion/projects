""" Candidate generation via distance in embedding space. """
from typing import Iterable

import numpy
from sklearn.neighbors import NearestNeighbors

from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span
from .base import NearestNeighborCandidateSelector
from rapidfuzz.string_metric import normalized_levenshtein


class CandidateSelector(NearestNeighborCandidateSelector):
    """Callable object selecting candidates as nearest neighbours in embedding space."""

    def _init_lookup_structure(
        self, kb: KnowledgeBase, max_n_candidates: int, **kwargs
    ) -> NearestNeighbors:
        container = NearestNeighbors(n_neighbors=max_n_candidates, metric="cosine")
        container.fit(
            numpy.asarray([kb.get_vector(ent_id) for ent_id in kb.get_entity_strings()])
        )

        return container

    def _fetch_candidates(
        self,
        dataset_id: str,
        span: Span,
        kb: KnowledgeBase,
        max_n_candidates: int,
        lexical_similarity_cutoff: float = 0.5,
    ) -> Iterable[int]:
        target_vec = self._pipelines[dataset_id](span.text).vector
        if not isinstance(target_vec, numpy.ndarray):
            target_vec = target_vec.get()

        nn_idx = self._lookup_structs[dataset_id].kneighbors(
            target_vec.reshape((1, -1))
        )[1][0]
        ent_ids = kb.get_entity_strings()
        nn_entities = {
            ent_ids[i]: self._entities[dataset_id][ent_ids[i]] for i in nn_idx
        }
        candidate_entity_ids = {
            match[0]
            for match in [
                (nne, normalized_levenshtein(name.lower(), span.text.lower()) / 100)
                for nne in nn_entities
                for name in nn_entities[nne].aliases
            ]
            if match[1] >= lexical_similarity_cutoff
        }

        return {
            cand
            for cands_for_alias in [
                kb.get_alias_candidates("_" + cei + "_") for cei in candidate_entity_ids
            ]
            for cand in cands_for_alias
        }
