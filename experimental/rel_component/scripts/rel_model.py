from typing import List, Tuple, Callable

import spacy
from spacy.tokens import Doc, Span
from thinc.types import Floats2d
from thinc.api import Model, Linear, chain, Logistic


@spacy.registry.architectures.register("rel_model.v1")
def create_relation_model(
    create_instance_tensor: Model[List[Doc], Floats2d],
    classification_layer: Model[Floats2d, Floats2d],
) -> Model[List[Doc], Floats2d]:
    with Model.define_operators({">>": chain}):
        model = create_instance_tensor >> classification_layer
        model.attrs["get_instances"] = create_instance_tensor.attrs["get_instances"]
    return model


@spacy.registry.architectures.register("rel_classification_layer.v1")
def create_classification_layer(
    nO: int = None, nI: int = None
) -> Model[Floats2d, Floats2d]:
    with Model.define_operators({">>": chain}):
        return Linear(nO=nO, nI=nI) >> Logistic()


@spacy.registry.misc.register("rel_instance_generator.v2")
def create_instances(max_length: int) -> Callable[[Doc], List[Tuple[Span, Span]]]:
    def get_instances(doc: Doc) -> List[Tuple[Span, Span]]:
        instances = []
        for ent1 in doc.ents:
            for ent2 in doc.ents:
                if ent1 != ent2:
                    if max_length and abs(ent2.start - ent1.start) <= max_length:
                        instances.append((ent1, ent2))
        return instances

    return get_instances


@spacy.registry.misc.register("rel_instance_tensor.v1")
def create_tensors(
    tok2vec: Model[List[Doc], List[Floats2d]],
    get_instances: Callable[[Doc], List[Tuple[Span, Span]]],
) -> Model[List[Doc], Floats2d]:

    return Model(
        "instance_tensors",
        instance_forward,
        layers=[tok2vec],
        refs={"tok2vec": tok2vec},
        attrs={"get_instances": get_instances},
        init=instance_init,
    )


def instance_forward(
    model: Model[List[Doc], Floats2d], docs: List[Doc], is_train: bool
) -> Tuple[Floats2d, Callable]:
    relations = []
    shapes = []
    instances = []
    tok2vec = model.get_ref("tok2vec")
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)
    get_instances = model.attrs["get_instances"]
    for doc, tokvec in zip(docs, tokvecs):
        ents = get_instances(doc)
        instances.append(ents)
        shapes.append(tokvec.shape)
        for (ent1, ent2) in ents:
            # take mean value of tokens within an entity
            v1 = tokvec[ent1.start : ent1.end].mean(axis=0)
            v2 = tokvec[ent2.start : ent2.end].mean(axis=0)
            relations.append(model.ops.xp.hstack((v1, v2)))

    def backprop(d_instances: Floats2d) -> List[Doc]:
        result = []
        d = 0
        for i, shape in enumerate(shapes):
            # TODO: make more efficient / succinct
            d_tokvecs = model.ops.alloc2f(*shape)
            row_dim = d_tokvecs.shape[1]
            ents = instances[i]
            indices = {}
            # collect all relevant indices for each entity
            for (ent1, ent2) in ents:
                t1 = (ent1.start, ent1.end)
                indices[t1] = indices.get(t1, [])
                indices[t1].append((d, 0, row_dim))

                t2 = (ent2.start, ent2.end)
                indices[t2] = indices.get(t2, [])
                indices[t2].append((d, row_dim, len(d_instances[d])))
                d += 1
            # for each entity, take the mean of its values
            for token, sources in indices.items():
                start, end = token
                for source in sources:
                    d_tokvecs[start:end] += d_instances[source[0]][
                        source[1] : source[2]
                    ]
                d_tokvecs[start:end] /= len(sources)
            result.append(d_tokvecs)
        return bp_tokvecs(result)

    return model.ops.asarray(relations), backprop


def instance_init(model: Model, X: List[Doc] = None, Y: Floats2d = None) -> Model:
    tok2vec = model.get_ref("tok2vec")
    if X is not None:
        tok2vec.initialize(X)
    return model
