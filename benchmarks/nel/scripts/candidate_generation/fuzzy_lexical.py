""" Candidate generation via distance in lexical space. """
from typing import Iterable

import numpy
from sklearn.neighbors import NearestNeighbors
from thefuzz import fuzz
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span
from .base import NearestNeighborCandidateSelector


class CandidateSelector(NearestNeighborCandidateSelector):
    """ Callable object selecting candidates as nearest neighbours in lexical space. """

    def _init_container(self, dataset_id: str, kb: KnowledgeBase, k: int) -> None:
        self._container[dataset_id] = NearestNeighbors(
            n_neighbors=k, algorithm="ball_tree", metric=lambda v, w: 100 - fuzz.ratio(v, w)
        )
        # todo @RM encode string numerically -
        #  https://stackoverflow.com/questions/227459/how-to-get-the-ascii-value-of-a-character
        self._container[dataset_id].fit(numpy.asarray([ent_id for ent_id in kb.get_entity_strings()]).reshape(-1, 1))

    def _fetch_candidates_idx(self, dataset_id: str, span: Span, kb: KnowledgeBase) -> Iterable[int]:
        return self._container[dataset_id].kneighbors(span.text)[1][0]
