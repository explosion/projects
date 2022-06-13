""" Base class generation for candidate selection. """
import pickle
from itertools import chain
from typing import Iterator, Dict, Any, Iterable

import numpy
from sklearn.neighbors import NearestNeighbors

import spacy
from spacy import Language
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span


# More elegant way to resolve import conflicts between training and evaluation calls?
try:
    from ..datasets.dataset import Dataset
except ValueError:
    from datasets.dataset import Dataset


class NearestNeighborCandidateSelector:
    """ Callable object selecting candidates via nearest neighbour search. """

    _pipelines: Dict[str, Language] = {}
    _container: Dict[str, NearestNeighbors] = {}
    _entities: Dict[str, Any] = {}

    def __call__(self, dataset_id: str, k: int, kb: KnowledgeBase, span: Span) -> Iterator[Candidate]:
        """ Identifies entity candidates.
        dataset_id (str): ID of dataset for which to select candidates.
        k (int): Numbers of nearest neighbours to query.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        span (Span): Span to match potential entity candidates with.
        RETURNS (Iterator[Candidate]): Candidates for specified entity.
        """

        if dataset_id not in self._pipelines:
            self._pipelines[dataset_id] = spacy.load(Dataset.assemble_paths(dataset_id)["nlp_base"])
            with open(Dataset.assemble_paths(dataset_id)["entities"], "rb") as file:
                self._entities[dataset_id] = pickle.load(file)
        if dataset_id not in self._container:
            self._init_container(dataset_id, kb, k)

        # Retrieve candidates from KB via their aliases.
        ent_ids = kb.get_entity_strings()
        return chain.from_iterable([
            kb.get_alias_candidates(next(iter(self._entities[dataset_id][ent_ids[i]]["names"])).replace("_", " "))
            for i in self._fetch_candidates_idx(dataset_id, span, kb)
        ])

    def _init_container(self, dataset_id: str, kb: KnowledgeBase, k: int) -> None:
        """ Init data structure for container.
        dataset_id (str): ID of dataset for which to select candidates.
        span (Span): candidate span.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        k (int): Numbers of nearest neighbours to query.
        """
        raise NotImplementedError

    def _fetch_candidates_idx(self, dataset_id: str, span: Span, kb: KnowledgeBase) -> Iterable[int]:
        """ Fetches candidate indices.
        dataset_id (str): ID of dataset for which to select candidates.
        span (Span): candidate span.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        RETURNS (Iterable[int]): Indices of candidates for specified entity.
        """
        raise NotImplementedError
