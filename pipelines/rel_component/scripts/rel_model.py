from typing import List, Callable, Tuple

import numpy
from thinc.types import Floats2d
from thinc.api import Model, Linear, Ops, Softmax

from spacy.util import registry


@registry.architectures.register("rel_model.v1")
def create_relation_model(
    tok2vec: Model[List["Doc"], Floats2d],
    create_candidate_tensor: Callable[[List["Doc"], Floats2d, Ops], Tuple[Floats2d, Callable]],
    get_candidates: Callable[["Doc"], List[Tuple["Span", "Span"]]],
    output_layer: Model[Floats2d, Floats2d],
    nO: int,
) -> Model[List["Doc"], Floats2d]:
    return Model(
        "relations",
        layers=[tok2vec, output_layer],
        refs={"tok2vec": tok2vec, "output_layer": output_layer},
        attrs={"create_candidate_tensor": create_candidate_tensor, "get_candidates": get_candidates},
        dims={"nO": nO},
        forward=forward,
        init=init,
    )


@registry.architectures.register("rel_cand_tensor.v1")
def create_candidates() -> Callable[[List["Doc"], Callable, Floats2d, Ops], Tuple[Floats2d, Callable]]:
    def get_candidate_tensor(docs: List["Doc"], get_candidates: Callable, tokvecs: Floats2d, ops: Ops):
        relations = []
        for i, doc in enumerate(docs):
            for (ent1, ent2) in get_candidates(doc):
                # take mean value of tokens within an entity
                v1 = tokvecs[i][ent1.start:ent1.end].mean(axis=0)
                v2 = tokvecs[i][ent2.start:ent2.end].mean(axis=0)
                relations.append(ops.xp.hstack((v1, v2)))
        with numpy.printoptions(precision=2, suppress=True):
            print(f"candidate data: {ops.asarray(relations)}")
        return ops.asarray(relations), lambda: None
    return get_candidate_tensor


@registry.architectures.register("rel_cand_generator.v1")
def create_candidate_indices() -> Callable[["Doc"], List[Tuple["Span", "Span"]]]:
    def get_candidate_indices(doc: "Doc"):
        indices = []
        for ent1 in doc.ents:
            for ent2 in doc.ents:
                indices.append((ent1, ent2))
        return indices
    return get_candidate_indices


@registry.architectures.register("rel_output_layer.v1")
def create_layer(nI: int = None, nO: int = None) -> Model[Floats2d, Floats2d]:
    # """This output layer currently assumes max. 1 REL label between two (directed) entities."""
    return Linear(nO=nO, nI=nI)
    # return Softmax(nO=nO, nI=nI)


def forward(model, docs, is_train):
    tok2vec = model.get_ref("tok2vec")
    get_candidates = model.attrs["get_candidates"]
    create_candidate_tensor = model.attrs["create_candidate_tensor"]
    output_layer = model.get_ref("output_layer")
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)
    cand_vectors, bp_cand = create_candidate_tensor(docs, get_candidates, tokvecs, model.ops)
    with numpy.printoptions(precision=2, suppress=True):
        print(" cand_vectors", cand_vectors)
    scores, bp_scores = output_layer(cand_vectors, is_train)
    with numpy.printoptions(precision=2, suppress=True):
        print(" scores", scores)

    def backprop(d_scores):
        return bp_tokvecs(bp_cand(bp_scores(d_scores)))

    return scores, backprop


def _make_candidate_vectors(ops, tokvecs, cand_ids):
    cand_vectors = ops.xp.vstack((tokvecs[cand_ids[:, 0]], tokvecs[cand_ids[:, 1]]))

    def backprop(d_candidates):
        d_tokvecs = ops.alloc2f(*tokvecs.shape)
        # Ugh, this is probably wrong. I hate scatter add :(
        ops.scatter_add(d_tokvecs, d_candidates, cand_ids)
        return d_tokvecs

    return cand_vectors, backprop


def init(
    model: Model, X: List["Doc"] = None, Y: Floats2d = None
) -> Model:
    tok2vec = model.get_ref("tok2vec")
    get_candidates = model.attrs["get_candidates"]
    create_candidate_tensor = model.attrs["create_candidate_tensor"]
    output_layer = model.get_ref("output_layer")
    if X is not None:
        tok2vec.initialize(X=X)
        tokvecs = tok2vec.predict(X)
        cand_vectors, _ = create_candidate_tensor(X, get_candidates, tokvecs, model.ops)
        output_layer.initialize(X=cand_vectors, Y=Y)
    return model
