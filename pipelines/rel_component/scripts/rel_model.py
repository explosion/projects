from typing import List, Callable
from thinc.types import Floats2d
from thinc.api import Model, Linear, Ops

from spacy.util import registry


@registry.architectures.register("my_rel_model.v1")
def create_relation_model(
    tok2vec: Model[List["Doc"], Floats2d],
    get_candidates: Callable[[Ops, List["Doc"], Floats2d], Floats2d],
    output_layer: Model[Floats2d, Floats2d],
    nO: int,
) -> Model:
    return Model(
        "relations",
        layers=[tok2vec, output_layer],
        refs={"tok2vec": tok2vec, "output_layer": output_layer},
        attrs={"get_candidates": get_candidates},
        dims={"nO": nO},
        forward=forward,
    )


@registry.architectures.register("my_rel_candidate_generator.v1")
def create_candidates() -> Callable[[Ops, List["Doc"], Floats2d], Floats2d]:
    def get_candidates(ops: Ops, docs: List["Doc"], tokvecs: Floats2d):
        print("docs", len(docs))
        print("tokvecs", tokvecs.shape)
        relations = []
        for doc in docs:
            for ent1 in doc.ents:
                for ent2 in doc.ents:
                    v1 = tokvecs[ent1.start:ent1.end]
                    print("v1", v1.shape)
                    v2 = tokvecs[ent2.start:ent2.end]
                    print("v2", v2.shape)
                    relations.append(ops.xp.vstack(v1, v2))
        return ops.asarray(relations)
    return get_candidates


@registry.architectures.register("my_rel_output_layer.v1")
def create_layer(nI, nO) -> Model[Floats2d, Floats2d]:
    return Linear(nO=nO, nI=nI)


def forward(model, docs, is_train):
    tok2vec = model.get_ref("tok2vec")
    get_candidates = model.attrs["get_candidates"]
    output_layer = model.get_ref("output_layer")
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)
    cand_ids = get_candidates(docs)
    candidates, bp_candidates = _make_candidate_vectors(model.ops, tokvecs, cand_ids)
    scores, bp_scores = output_layer(candidates, is_train)

    def backprop(d_scores):
        return bp_tokvecs(bp_candidates(bp_scores(d_scores)))

    return scores, backprop


def _make_candidate_vectors(ops, tokvecs, cand_ids):
    candidates = ops.xp.vstack((tokvecs[cand_ids[:, 0]], tokvecs[cand_ids[:, 1]]))

    def backprop(d_candidates):
        d_tokvecs = ops.alloc2f(*tokvecs.shape)
        # Ugh, this is probably wrong. I hate scatter add :(
        ops.scatter_add(d_tokvecs, d_candidates, cand_ids)
        return d_tokvecs

    return candidates, backprop
