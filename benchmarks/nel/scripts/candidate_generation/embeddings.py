""" Candidate generation via distance in embedding space. """
from typing import Iterable

import numpy
from sklearn.neighbors import NearestNeighbors
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span
from .base import NearestNeighborCandidateSelector


class CandidateSelector(NearestNeighborCandidateSelector):
    """ Callable object selecting candidates as nearest neighbours in embedding space. """

    def _init_container(self, dataset_id: str, kb: KnowledgeBase, k: int) -> None:
        self._container[dataset_id] = NearestNeighbors(n_neighbors=k, metric="cosine")
        self._container[dataset_id].fit(numpy.asarray([kb.get_vector(ent_id) for ent_id in kb.get_entity_strings()]))

    def _fetch_candidates_idx(self, dataset_id: str, span: Span, kb: KnowledgeBase) -> Iterable[int]:
        target_vec = self._pipelines[dataset_id](span.text).vector
        if not isinstance(target_vec, numpy.ndarray):
            target_vec = target_vec.get()

        return self._container[dataset_id].kneighbors(target_vec.reshape((1, -1)))[1][0]
