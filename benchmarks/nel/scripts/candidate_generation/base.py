""" Base class generation for candidate selection. """
import pickle
from itertools import chain
from typing import Iterator, Dict, Any, Iterable, Optional

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
    _container: Dict[str, Any] = {}
    _entities: Dict[str, Any] = {}

    def __call__(self, kb: KnowledgeBase, span: Span, dataset_id: str, max_n_candidates: int, **kwargs) -> Iterator[Candidate]:
        """ Identifies entity candidates.
        dataset_id (str): ID of dataset for which to select candidates.
        max_n_candidates (int): Numbers of nearest neighbours to query.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        span (Span): Span to match potential entity candidates with.
        RETURNS (Iterator[Candidate]): Candidates for specified entity.
        """

        if dataset_id not in self._pipelines:
            self._pipelines[dataset_id] = spacy.load(Dataset.assemble_paths(dataset_id)["nlp_base"])
            with open(Dataset.assemble_paths(dataset_id)["entities"], "rb") as file:
                self._entities[dataset_id] = pickle.load(file)
        if dataset_id not in self._container:
            self._container[dataset_id] = self._init_container(dataset_id, kb, max_n_candidates, **kwargs)

        # Retrieve candidates from KB via their aliases.
        return self._fetch_candidates(dataset_id, span, kb, max_n_candidates, **kwargs)

    def _init_container(self, dataset_id: str, kb: KnowledgeBase, max_n_candidates: int, **kwargs) -> Any:
        """ Init data structure for container.
        dataset_id (str): ID of dataset for which to select candidates.
        span (Span): candidate span.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        max_n_candidates (int): Max. number of candidates to generate.
        RETURNS (Any): Initialized container.
        """
        raise NotImplementedError

    def _fetch_candidates(
        self, dataset_id: str, span: Span, kb: KnowledgeBase, max_n_candidates: int, **kwargs
    ) -> Iterator[Candidate]:
        """ Fetches candidates for entity in span.text.
        dataset_id (str): ID of dataset for which to select candidates.
        span (Span): candidate span.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        max_n_candidates (int): Max. number of candidates to generate.
        RETURNS (Iterator[Candidate]): Candidates for specified entity.
        """
        raise NotImplementedError
