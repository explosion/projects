from thinc.api import chain, concatenate, Maxout
from thinc.api import reduce_mean, reduce_max
from spacy.util import registry
from .reduce_last import reduce_last
from .reduce_first import reduce_first


@registry.architectures.register("ConcatReducer.v1")
def build_concat_reducer(hidden_size: int):
    return chain(
        concatenate(reduce_last(), reduce_first(), reduce_mean(), reduce_max()),
        Maxout(nO=hidden_size, normalize=True, dropout=0.0)
    )
