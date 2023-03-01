""" Base class generation for candidate selection. """
import abc
import pickle
from typing import Dict, Any, Optional, Iterable, Tuple

import spacy
from spacy import Language
from spacy.kb import Candidate
from spacy.tokens import Span

from ..compat import KnowledgeBase
from datasets.dataset import Dataset


class NearestNeighborCandidateSelector(abc.ABC):
    """Callable object selecting candidates via nearest neighbour search."""

    _pipeline: Optional[Language] = None
    _lookup_struct: Optional[Any] = None
    _entities: Dict[str, Any] = {}

    def __call__(
        self, kb: KnowledgeBase, span: Span, dataset_id: str, language: str, max_n_candidates: int, **kwargs
    ) -> Iterable[Candidate]:
        """Identifies entity candidates.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        span (Span): Span to match potential entity candidates with.
        dataset_id (str): ID of dataset for which to select candidates.
        language (str): Language.
        max_n_candidates (int): Numbers of nearest neighbours to query.
        RETURNS (Iterator[Candidate]): Candidates for specified entity.
        """

        if self._pipeline is None:
            # Load pipeline and pickled entities. Run name doesn't matter for either of those.
            paths = Dataset.assemble_paths(dataset_id, "", language)
            self._pipeline = spacy.load(paths["nlp_base"])
            with open(paths["entities"], "rb") as file:
                self._entities[dataset_id] = pickle.load(file)
        if self._lookup_struct is None:
            self._lookup_struct = self._init_lookup_structure(kb, max_n_candidates, **kwargs)

        # Retrieve candidates from KB.
        return self._fetch_candidates(dataset_id, span, kb, max_n_candidates, **kwargs)

    @abc.abstractmethod
    def _init_lookup_structure(self, kb: KnowledgeBase, max_n_candidates: int, **kwargs) -> Any:
        """Init container for lookups for new dataset. Doesn't do anything if initialized for this dataset already.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        max_n_candidates (int): Max. number of candidates to generate.
        RETURNS (Any): Initialized container.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _fetch_candidates(
        self,
        dataset_id: str,
        span: Span,
        kb: KnowledgeBase,
        max_n_candidates: int,
        **kwargs
    ) -> Iterable[Candidate]:
        """Fetches candidates for entity in span.text.
        dataset_id (str): ID of dataset for which to select candidates.
        span (Span): candidate span.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        max_n_candidates (int): Max. number of candidates to generate.
        RETURNS (Iterator[Candidate]): Candidates for specified entity.
        """
        raise NotImplementedError
