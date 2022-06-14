""" Candidate generation via distance in embedding space. """
from itertools import chain
from typing import Iterable

import numpy
from sklearn.neighbors import NearestNeighbors
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span
from .base import NearestNeighborCandidateSelector


class CandidateSelector(NearestNeighborCandidateSelector):
    """ Callable object selecting candidates as nearest neighbours in embedding space. """

    def _init_container(self, dataset_id: str, kb: KnowledgeBase, max_n_candidates: int, **kwargs) -> NearestNeighbors:
        container = NearestNeighbors(n_neighbors=max_n_candidates, metric="cosine")
        container.fit(numpy.asarray([kb.get_vector(ent_id) for ent_id in kb.get_entity_strings()]))

        return container

    def _fetch_candidates(
        self, dataset_id: str, span: Span, kb: KnowledgeBase, max_n_candidates: int, **kwargs
    ) -> Iterable[int]:
        target_vec = self._pipelines[dataset_id](span.text).vector
        if not isinstance(target_vec, numpy.ndarray):
            target_vec = target_vec.get()

        ent_ids = kb.get_entity_strings()
        return chain.from_iterable([
            kb.get_alias_candidates(next(iter(self._entities[dataset_id][ent_ids[i]]["names"])).replace("_", " "))
            for i in self._container[dataset_id].kneighbors(target_vec.reshape((1, -1)))[1][0]
        ])
