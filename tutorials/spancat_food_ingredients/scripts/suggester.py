from spacy import registry
from typing import List, Optional, Iterable, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d
from spacy.tokens import Doc
from spacy.util import registry
from spacy.pipeline.spancat import Suggester


@registry.misc("noun_suggester")
def build_noun_suggester():
    """Suggester that suggests every token and token sequence which .pos_ attribute is a NOUN"""

    def noun_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()

        spans = []
        lengths = []

        for doc in docs:
            cache = set()
            length = 0

            for token in doc:
                if token.pos_ == "NOUN":
                    start_i = token.i
                    end_i = token.i
                    if (start_i, end_i + 1) not in cache:
                        spans.append((start_i, end_i + 1))
                        cache.add((start_i, end_i + 1))
                        length += 1
                    for token_k in doc[token.i :]:
                        if token_k.pos_ == "NOUN":
                            end_i = token_k.i
                            if (start_i, end_i + 1) not in cache:
                                spans.append((start_i, end_i + 1))
                                cache.add((start_i, end_i + 1))
                                length += 1
                        else:
                            break

            lengths.append(length)

        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        if len(spans) > 0:
            output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

        return output

    return noun_suggester
