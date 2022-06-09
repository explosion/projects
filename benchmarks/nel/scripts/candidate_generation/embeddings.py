""" Candidate generation via distance in embedding space. """
from typing import Iterator, Dict

import numpy
import spacy
from spacy import Language
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span

from scripts.datasets.dataset import Dataset
from sklearn.neighbors import NearestNeighbors


class CandidateSelector:
    """ Callable object selecting candidates as nearest neighbours in embedding space.
    Includes primitive caching.
    """

    _pipelines: Dict[str, Language] = {}
    _kdtree: Dict[str, NearestNeighbors] = {}

    def __call__(self, dataset_id: str, k: int, kb: KnowledgeBase, span: Span) -> Iterator[Candidate]:
        """ Identifies entity candidates via their embeddings.
        dataset_id (str): ID of dataset for which to select candidates.
        k (int): Numbers of nearest neighbours to query.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        span (Span): Span to match potential entity candidates with.
        """

        # todo @RM
        #  - exchange
        #  - define offcut/top-n in config, return relevant candidates
        #  - evaluate results
        #  - draft postprocessing steps
        #  - compare with get_candidates()
        if dataset_id not in self._pipelines:
            self._pipelines[dataset_id] = spacy.load(Dataset.assemble_paths(dataset_id)["nlp_base"])
        if dataset_id not in self._kdtree:
            self._kdtree[dataset_id] = NearestNeighbors(n_neighbors=k, metric="cosine")
            self._kdtree[dataset_id].fit(numpy.asarray([kb.get_vector(eid) for eid in kb.get_entity_strings()]))

        # Fetch n nearest neighbors.
        target = self._pipelines[dataset_id](span.text).vector
        if not isinstance(target, numpy.ndarray):
            target = target.get()
        nn_idx = self._kdtree[dataset_id].kneighbors(target.reshape((1, -1)))  # type: ignore

        return kb.get_alias_candidates(span.text)
