from typing import List, Tuple, Callable
from thinc.api import Model, chain
from thinc.api import Maxout
from thinc.types import Floats2d

import numpy as np

from spacy.util import registry
from spacy.tokens import Doc


@registry.architectures("spacy.SpanBoundaryDetection.v1")
def build_boundary_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    featurer: Model[List[Floats2d], Floats2d],
    scorer: Model[Floats2d, Floats2d],
    hidden_size: int,
) -> Model[List[Doc], Floats2d]:

    model = chain(
        tok2vec, featurer, Maxout(nO=hidden_size, normalize=True, dropout=0.0), scorer
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("featurer", featurer)
    model.set_ref("scorer", scorer)
    return model


@registry.layers("spacy.SpanBoundaryFeaturer.v1")
def feature_vectors(window_size: int) -> Model[List[Floats2d], Floats2d]:
    def init(model, X=None, Y=None):
        pass

    def forward(
        model: Model, docs: List[Floats2d], is_train: bool
    ) -> Tuple[Floats2d, Callable]:

        modified_vectors = []
        # Iterate over docs
        for doc in docs:
            # Iterate over token vectors
            for i, token_vector in enumerate(doc):
                window_vectors = []
                _min = window_size
                _max = window_size

                if i + _max >= len(doc):
                    _max = (len(doc) - i) - 1

                if i - _min < 0:
                    _min = i

                for k in range(i - _min, i + _max + 1):
                    window_vectors.append(doc[k])

                # Calculate features
                max_vector = model.ops.reduce_max(window_vectors, axis=0)
                mean_vector = model.ops.reduce_mean(window_vectors, axis=0)
                modified_vector = model.ops.asarray(
                    model.ops.xp.concatenate([token_vector, mean_vector, max_vector])
                )

                # Add to list
                modified_vectors.append(modified_vector)

        def backprop(_vectors: Floats2d) -> Floats2d:
            print(f"Backprop: {type(_vectors)} {_vectors.shape}")
            #return _vectors
            #return model.ops.unflatten(_vectors.data, _vectors.lengths)
            return []

        return model.ops.asarray(modified_vectors), backprop

    return Model(
        "feature_vectors", forward, layers=[], refs={}, attrs={}, dims={}, init=init
    )
