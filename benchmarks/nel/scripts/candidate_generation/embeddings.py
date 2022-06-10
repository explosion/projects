""" Candidate generation via distance in embedding space. """
import pickle
from itertools import chain
from typing import Iterator, Dict, Any

import numpy
import spacy
from spacy import Language
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span

# More elegant way to resolve import conflicts between training and evaluation calls?
try:
    from ..datasets.dataset import Dataset
except ValueError:
    from datasets.dataset import Dataset


from sklearn.neighbors import NearestNeighbors


class CandidateSelector:
    """ Callable object selecting candidates as nearest neighbours in embedding space. """

    _pipelines: Dict[str, Language] = {}
    _kdtree: Dict[str, NearestNeighbors] = {}
    _entities: Dict[str, Any] = {}

    def __call__(self, dataset_id: str, k: int, kb: KnowledgeBase, span: Span) -> Iterator[Candidate]:
        """ Identifies entity candidates via their embeddings.
        dataset_id (str): ID of dataset for which to select candidates.
        k (int): Numbers of nearest neighbours to query.
        kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
        span (Span): Span to match potential entity candidates with.
        """

        ent_ids = kb.get_entity_strings()

        # todo @RM
        #  - draft postprocessing steps
        #  - compare with get_candidates()
        #  - post summary on slack

        if dataset_id not in self._pipelines:
            self._pipelines[dataset_id] = spacy.load(Dataset.assemble_paths(dataset_id)["nlp_base"])
            with open(Dataset.assemble_paths(dataset_id)["entities"], "rb") as file:
                self._entities[dataset_id] = pickle.load(file)
        if dataset_id not in self._kdtree:
            self._kdtree[dataset_id] = NearestNeighbors(n_neighbors=k, metric="cosine")
            self._kdtree[dataset_id].fit(numpy.asarray([kb.get_vector(ent_id) for ent_id in ent_ids]))

        # Fetch n nearest neighbors.
        target_vec = self._pipelines[dataset_id](span.text).vector
        if not isinstance(target_vec, numpy.ndarray):
            target_vec = target_vec.get()
        _, nn_idx = self._kdtree[dataset_id].kneighbors(target_vec.reshape((1, -1)))

        return [
            *kb.get_alias_candidates(span.text),
            # Retrieve candidates from KB via their aliases, flatten 2D list with chain.from_iterable().
            *chain.from_iterable([
                kb.get_alias_candidates(next(iter(self._entities[dataset_id][ent_ids[i]]["names"])).replace("_", " "))
                for i in nn_idx[0]
            ])
        ]
