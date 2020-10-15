from typing import List, Callable, Tuple, Optional

import numpy
from thinc.types import Floats2d
from thinc.api import Model, Linear, Ops, Softmax

from spacy.util import registry


@registry.architectures.register("rel_model.v1")
def create_relation_model(
    tok2vec: Model[List["Doc"], List[Floats2d]],
    get_candidates: Callable[["Doc"], List[Tuple["Span", "Span"]]],
    create_candidate_tensor: Callable[[List["Doc"], Floats2d, Ops], Tuple[Floats2d, Callable]],
    output_layer: Model[Floats2d, Floats2d],
    nO: Optional[int],
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


@registry.misc.register("rel_cand_tensor.v1")
def create_tensors() -> Callable[[List["Doc"], Callable, List[Floats2d], Ops], Tuple[Floats2d, Callable]]:
    def get_candidate_tensor(docs: List["Doc"], get_candidates: Callable, tokvecs: List[Floats2d], ops: Ops):
        with numpy.printoptions(precision=2, suppress=True):
            print()
            print(f"get candidate tensor, tokvecs {tokvecs}")
        relations = []
        shapes = []
        candidates = []
        for i, doc in enumerate(docs):
            ents = get_candidates(doc)
            candidates.append(ents)
            shapes.append(tokvecs[i].shape)
            for (ent1, ent2) in ents:
                # take mean value of tokens within an entity
                v1 = tokvecs[i][ent1.start:ent1.end].mean(axis=0)
                v2 = tokvecs[i][ent2.start:ent2.end].mean(axis=0)
                relations.append(ops.xp.hstack((v1, v2)))
        with numpy.printoptions(precision=2, suppress=True):
            print(f"candidate data: {ops.asarray(relations)}")
            print("shapes", shapes)

        def backprop(d_candidates):
            with numpy.printoptions(precision=2, suppress=True):
                print(f"calling backprop for: {d_candidates} {type(d_candidates)}")
            result = []
            d = 0
            for i, shape in enumerate(shapes):
                # TODO: make more efficient
                d_tokvecs = ops.alloc2f(*shape)
                row_dim = d_tokvecs.shape[1]
                ents = candidates[i]
                indices = {}
                with numpy.printoptions(precision=2, suppress=True):
                    # print()
                    for (ent1, ent2) in ents:
                        t1 = (ent1.start, ent1.end)
                        indices[t1] = indices.get(t1, [])
                        indices[t1].append((d, 0, row_dim))

                        t2 = (ent2.start, ent2.end)
                        indices[t2] = indices.get(t2, [])
                        indices[t2].append((d, row_dim, len(d_candidates[d])))
                        d += 1
                    for token, sources in indices.items():
                        start, end = token
                        for source in sources:
                            d_tokvecs[start:end] += d_candidates[source[0]][source[1]:source[2]]
                            # print("token", start, token[1], "-->", d_tokvecs[start:end])
                        d_tokvecs[start:end] /= len(sources)

                    # print(i, "d_tokvecs", d_tokvecs)
                result.append(d_tokvecs)
            with numpy.printoptions(precision=2, suppress=True):
                print("result", result)
            return result

        return ops.asarray(relations), backprop
    return get_candidate_tensor


@registry.misc.register("rel_cand_generator.v2")
def create_candidates(max_length: int) -> Callable[["Doc"], List[Tuple["Span", "Span"]]]:
    def get_candidates(doc: "Doc") -> List[Tuple["Span", "Span"]]:
        candidates = []
        for ent1 in doc.ents:
            for ent2 in doc.ents:
                if ent1 != ent2:
                    if max_length and abs(ent2.start - ent1.start) <= max_length:
                        candidates.append((ent1, ent2))
        return candidates
    return get_candidates


@registry.architectures.register("rel_output_layer.v1")
def create_layer(nI: int = None, nO: int = None) -> Model[Floats2d, Floats2d]:
    return Linear(nO=nO, nI=nI)


def forward(model, docs, is_train):
    print()
    print("FORWARD")
    tok2vec = model.get_ref("tok2vec")
    get_candidates = model.attrs["get_candidates"]
    create_candidate_tensor = model.attrs["create_candidate_tensor"]
    output_layer = model.get_ref("output_layer")
    for doc in docs:
        print(doc.text)
        print(tok2vec([doc], is_train=False))
    print()
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)
    print("tokvecs", tokvecs)
    cand_vectors, bp_cand = create_candidate_tensor(docs, get_candidates, tokvecs, model.ops)
    with numpy.printoptions(precision=2, suppress=True):
        print(" cand_vectors", cand_vectors)
    scores, bp_scores = output_layer(cand_vectors, is_train)
    with numpy.printoptions(precision=2, suppress=True):
        print(" scores", scores)

    def backprop(d_scores):
        return bp_tokvecs(bp_cand(bp_scores(d_scores)))

    return scores, backprop


def init(
    model: Model, X: List["Doc"] = None, Y: Floats2d = None
) -> Model:
    print()
    print("INIT rel_model")
    print("before")
    get_candidates = model.attrs["get_candidates"]
    create_candidate_tensor = model.attrs["create_candidate_tensor"]
    output_layer = model.get_ref("output_layer")
    tok2vec = model.get_ref("tok2vec")
    if X is not None:
        tok2vec.initialize(X=X)
        for doc in X:
            print(doc.text)
            print(tok2vec([doc], is_train=False))
        tokvecs = tok2vec.predict(X)
        cand_vectors, _ = create_candidate_tensor(X, get_candidates, tokvecs, model.ops)
        output_layer.initialize(X=cand_vectors, Y=Y)

    if model.has_dim("nO") is None:
        model.set_dim("nO", output_layer.get_dim("nO"))
    print("after")
    for doc in X:
        print(doc.text)
        print(tok2vec([doc], is_train=False))
    return model
