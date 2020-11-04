from typing import List, Tuple, Callable

import numpy
import spacy
from spacy.tokens import Doc, Span
from thinc.types import Floats2d, Ints1d, Ragged, cast
from thinc.api import Model, Linear, chain, Logistic

DEBUG = False


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
    pooling: Model[Ragged, Floats2d],
    get_instances: Callable[[Doc], List[Tuple[Span, Span]]],
) -> Model[List[Doc], Floats2d]:

    return Model(
        "instance_tensors",
        instance_forward,
        layers=[tok2vec, pooling],
        refs={"tok2vec": tok2vec, "pooling": pooling},
        attrs={"get_instances": get_instances},
        init=instance_init,
    )


def instance_forward(
    model: Model[List[Doc], Floats2d], docs: List[Doc], is_train: bool
) -> Tuple[Floats2d, Callable]:
    if DEBUG:
        print()
        print("instance forward")
        print("docs", docs)
        for doc in docs:
            for ent in doc.ents:
                print(ent.text, ent.start, ent.end)

    pooling: Model[Ragged, Floats2d] = model.get_ref("pooling")
    tok2vec: Model[List[Doc], List[Floats2d]] = model.get_ref("tok2vec")
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)
    get_instances = model.attrs["get_instances"]
    all_instances = [get_instances(doc) for doc in docs]
    ents = []
    lengths = []

    with numpy.printoptions(precision=2, suppress=True):
        for doc_nr, (instances, tokvec) in enumerate(zip(all_instances, tokvecs)):
            if DEBUG:
                print()
                print("doc", doc_nr)
                print("tokvec", tokvec)
            token_indices = []
            for instance in instances:
                for ent in instance:
                    token_indices.extend([i for i in range(ent.start, ent.end)])
                    lengths.append(ent.end - ent.start)
            entity_array = tokvec[token_indices]
            ents.append(entity_array)
        array = model.ops.flatten(ents)

        if DEBUG:
            print("DONE")
            print("ents", ents)
            print("array", array)
            print("lenghts", lengths)

        entities = Ragged(array, cast(Ints1d, model.ops.asarray(lengths, dtype="int32")))
        pooled, bp_pooled = pooling(entities, is_train)

        # Reshape so that pairs of rows are concatenated.
        relations = model.ops.reshape2f(pooled, -1, pooled.shape[1] * 2)
        if DEBUG:
            print("entities", entities)
            print("pooled", pooled)
            print("relations", relations.shape)
            print(relations)
            print()

    def backprop(d_relations: Floats2d) -> List[Doc]:
        with numpy.printoptions(precision=2, suppress=True):
            if DEBUG:
                print()
                print("instance backprop")
                print("d_relations", d_relations.shape)
                print(d_relations)

            d_pooled = model.ops.reshape2f(d_relations, d_relations.shape[0] * 2, -1)
            d_ents = bp_pooled(d_pooled).data
            d_tokvecs = []
            t = 0
            if DEBUG:
                print("d_pooled", d_pooled)
                print("d_ents", d_ents)
            for doc_nr, instances in enumerate(all_instances):
                shape = tokvecs[doc_nr].shape
                d_tokvec = model.ops.alloc2f(*shape)
                count_occ = model.ops.alloc2f(*shape)
                if DEBUG:
                    print("doc_nr", doc_nr)
                    print("shape", shape)
                    print("d_tokvec", d_tokvec)
                for instance in instances:
                    for ent in instance:
                        d_tokvec[ent.start: ent.end] += d_ents[t]
                        count_occ[ent.start: ent.end] += 1
                        t += (ent.end - ent.start)
                d_tokvec /= (count_occ+0.00000000001)
                d_tokvecs.append(d_tokvec)
                if DEBUG:
                    print(" --> d_tokvec", d_tokvec)

            if DEBUG:
                print(" ---> d_tokvecs", d_tokvecs)
            d_docs = bp_tokvecs(d_tokvecs)
        return d_docs

    return model.ops.asarray(relations), backprop


def instance_init(model: Model, X: List[Doc] = None, Y: Floats2d = None) -> Model:
    tok2vec = model.get_ref("tok2vec")
    if X is not None:
        tok2vec.initialize(X)
    return model
