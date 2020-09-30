from typing import List, Callable, Tuple
from thinc.types import Floats2d
from thinc.api import Model, Linear, Ops

from spacy.util import registry


@registry.architectures.register("my_rel_model.v1")
def create_relation_model(
    tok2vec: Model[List["Doc"], Floats2d],
    get_candidates: Callable[[List["Doc"], Floats2d, Ops], Tuple[Floats2d, Callable]],
    output_layer: Model[Floats2d, Floats2d],
    nO: int,
) -> Model[List["Doc"], Floats2d]:
    return Model(
        "relations",
        layers=[tok2vec, output_layer],
        refs={"tok2vec": tok2vec, "output_layer": output_layer},
        attrs={"get_candidates": get_candidates},
        dims={"nO": nO},
        forward=forward,
        init=init,
    )


@registry.architectures.register("my_rel_candidate_generator.v1")
def create_candidates() -> Callable[[List["Doc"], Floats2d, Ops], Tuple[Floats2d, Callable]]:
    def get_candidates(docs: List["Doc"], tokvecs: Floats2d, ops: Ops):
        print("docs", len(docs))
        print("tokvecs", len(tokvecs))
        print("tokvecs", tokvecs)
        relations = []
        for i, doc in enumerate(docs):
            for ent1 in doc.ents:
                for ent2 in doc.ents:
                    # take mean value of tokens within an entity
                    v1 = tokvecs[i][ent1.start:ent1.end].mean(axis=0)
                    v2 = tokvecs[i][ent2.start:ent2.end].mean(axis=0)
                    relations.append(ops.xp.hstack((v1, v2)))
        print("relations", ops.asarray(relations))
        return ops.asarray(relations), None
    return get_candidates


@registry.architectures.register("my_rel_output_layer.v1")
def create_layer(nI, nO) -> Model[Floats2d, Floats2d]:
    return Linear(nO=nO, nI=nI)


def forward(model, docs, is_train):
    tok2vec = model.get_ref("tok2vec")
    get_candidates = model.attrs["get_candidates"]
    output_layer = model.get_ref("output_layer")
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)
    cand_vectors, bp_cand = get_candidates(docs, tokvecs, model.ops)
    scores, bp_scores = output_layer(cand_vectors, is_train)

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
    output_layer = model.get_ref("output_layer")
    if X is not None:
        tok2vec.initialize(X=X)
        tokvecs = tok2vec.predict(X)
        cand_vectors, _ = get_candidates(X, tokvecs, model.ops)
        output_layer.initialize(X=cand_vectors, Y=Y)
    return model
