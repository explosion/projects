from pathlib import Path

import pytest
from spacy.cli.project.assets import project_assets
from spacy.cli.project.run import project_run

from scripts.preprocess import _get_contiguous_tokens


@pytest.mark.parametrize(
    "labels,span",
    [
        ([0, 1, 1, 1, 0, 0], [(1, 3)]),
        ([0, 0, 0, 0, 1, 0], [(4,)]),
        ([0, 1, 1, 0, 0, 1], [(1, 2), (5,)]),
    ],
)
def test_contiguous_tokens_generator(labels, span):
    """Test if the method for generating contiguous tokens from EBM-NLP works correctly"""
    assert set(_get_contiguous_tokens(labels)) == set(span)


def test_ner_spancat_compare_project():
    root = Path(__file__).parent
    project_assets(root)
    # TODO: Uncomment once workflow has been implemented
    # project_run(root, "all", capture=True)
