from typing import List, Optional, Tuple, Callable

from spacy.tokens.doc import Doc
from thinc.types import Floats2d
from thinc.api import Model, Linear, chain, Logistic

from spacy.util import registry


@registry.architectures.register("rel_model.v1")
def create_relation_model(
    create_candidate_tensor: Model[List[Doc], Floats2d],
    classification_layer: Model[Floats2d, Floats2d],
    nO: Optional[int],
) -> Model[List[Doc], Floats2d]:
    return Model(
        "relations",
        layers=[create_candidate_tensor, classification_layer],
        refs={"create_candidate_tensor": create_candidate_tensor, "classification": classification_layer},
        dims={"nO": nO},
        forward=forward,
        init=init,
        attrs={
            "get_candidates": create_candidate_tensor.attrs["get_candidates"],
        },
    )


@registry.architectures.register("rel_classification_layer.v1")
def create_classification_layer(nI: int = None, nO: int = None) -> Model[Floats2d, Floats2d]:
    with Model.define_operators({">>": chain}):
        return Linear(nO=nO, nI=nI) >> Logistic()


def forward(model: Model[List[Doc], Floats2d], docs: List[Doc], is_train) -> Tuple[Floats2d, Callable]:
    create_candidate_tensor = model.get_ref("create_candidate_tensor")
    classification_layer = model.get_ref("classification")

    cand_vectors, bp_cand = create_candidate_tensor(docs, is_train)
    scores, bp_scores = classification_layer(cand_vectors, is_train)

    def backprop(d_scores: Floats2d) -> List[Doc]:
        return bp_cand(bp_scores(d_scores))

    return scores, backprop


def init(model: Model, X: List[Doc] = None, Y: Floats2d = None) -> Model:
    create_candidate_tensor = model.get_ref("create_candidate_tensor")
    classification_layer = model.get_ref("classification")
    if X is not None:
        create_candidate_tensor.initialize(X=X)
        cand_vectors = create_candidate_tensor.predict(X)
        classification_layer.initialize(X=cand_vectors, Y=Y)

    if model.has_dim("nO") is None:
        model.set_dim("nO", classification_layer.get_dim("nO"))
    return model
