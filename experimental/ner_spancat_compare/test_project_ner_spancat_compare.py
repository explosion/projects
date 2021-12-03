import pytest
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
