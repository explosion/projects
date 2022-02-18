import numpy
from typing import List, Dict, Callable, Tuple, Optional, Iterable, Any, cast
from thinc.api import Config, Model, get_current_ops, set_dropout_rate, Ops
from thinc.api import Optimizer
from thinc.types import Ragged, Ints2d, Floats2d, Ints1d

from spacy.compat import Protocol, runtime_checkable
from spacy.scorer import Scorer
from spacy.language import Language
from spacy.tokens import Doc, SpanGroup, Span
from spacy.vocab import Vocab
from spacy.training import Example, validate_examples
from spacy.errors import Errors
from spacy.util import registry

@runtime_checkable
class Suggester(Protocol):
    def __call__(self, docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        ...


@registry.misc("spacy.subtree_suggester.v1")
def build_subtree_suggester() -> Suggester:
    """Suggest every connected subtree per token"""

    def subtree_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        sizes = [1,2]
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []
        for doc in docs:
            starts = ops.xp.arange(len(doc), dtype="i")
            starts = starts.reshape((-1, 1))
            length = 0
            for size in sizes:
                if size <= len(doc):
                    starts_size = starts[: len(doc) - (size - 1)]
                    spans.append(ops.xp.hstack((starts_size, starts_size + size)))
                    length += spans[-1].shape[0]
                if spans:
                    assert spans[-1].ndim == 2, spans[-1].shape
            lengths.append(length)
        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        
        if len(spans) > 0:
            output = Ragged(ops.xp.vstack(spans), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

        assert output.dataXd.ndim == 2
        return output

    return subtree_suggester

if __name__ == "__main__":

    import spacy

    nlp = spacy.blank("en")
    text = "This is awesome"
    doc = nlp(text)
    suggester = build_subtree_suggester()

    candidate_indices = suggester([doc])

    print(candidate_indices)