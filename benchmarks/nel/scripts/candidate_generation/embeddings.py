""" Candidate generation. """
from pathlib import Path
from typing import Iterator, Dict

import spacy
from spacy import Language
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span

from scripts.datasets.dataset import Dataset


_pipelines: Dict[Path, Language] = {}


def create_candidates(dataset_name: str, kb: KnowledgeBase, span: Span) -> Iterator[Candidate]:
    """ Identifies entity candidates via their embeddings.
    kb (KnowledgeBase): KnowledgeBase containing all possible entity candidates.
    span (Span): Span to match potential entity candidates with.
    """

    # todo @RM
    #  - add temp getters to KB class to access all information necessary to create candidate objects
    #  - infer vector for span.text
    #  - compute distances between KB entity vectors and span.text vector
    #  - define offcut/top-n in config, return relevant candidates
    #  - evaluate results, compare with get_candidates()
    # nlp_path = Dataset.assemble_paths(dataset_id)["nlp_base"]
    # if nlp_path not in _pipelines:
    #     _pipelines[nlp_path] = spacy.load(nlp_path)

    return kb.get_alias_candidates(span.text)
