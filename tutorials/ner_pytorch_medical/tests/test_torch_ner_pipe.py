from re import T
from typing import Dict, List, Tuple
import pytest
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.vocab import Vocab
from spacy import util
from scripts.custom_functions import make_torch_entity_recognizer


TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]

PARTIAL_TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("", {"entities": []}),
]


def examples_from_annotations(annotations: List[Tuple[str, Dict[str, Tuple[int, int, str]]]]):
    nlp = English()
    examples = []
    for t in annotations:
        examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    return examples


@pytest.mark.parametrize("train_data", [TRAIN_DATA, PARTIAL_TRAIN_DATA])
def test_train(train_data):
    """Test that training succeeds and empty text does not throw errors."""
    nlp = English()
    train_examples = examples_from_annotations(train_data)
    nlp.add_pipe("torch_ner", last=True)
    nlp.initialize(lambda: train_examples)
    for _ in range(2):
        losses = {}
        batches = util.minibatch(train_examples, size=8)
        for batch in batches:
            nlp.update(batch, losses=losses)


def test_torch_ner_predict():
    """Test the prediction can handle empty docs"""
    nlp = spacy.blank("en")
    torch_ner = nlp.add_pipe("torch_ner")

    torch_ner.add_label("B")
    torch_ner.add_label("I")
    torch_ner.add_label("O")

    nlp.initialize()

    empty_doc = Doc(nlp.vocab)
    doc = nlp.make_doc("Test doc.")
    
    scores = torch_ner.predict([empty_doc])
    torch_ner.set_annotations([empty_doc], scores)

    empty_doc_preds = torch_ner(empty_doc)
    assert len(empty_doc_preds) == len(Doc(nlp.vocab))

    empty_preds = list(torch_ner.pipe([]))
    assert empty_preds == []

    docs = [empty_doc, doc]

    empty_example = Example(empty_doc, empty_doc)
    example = Example(doc, doc)

    nlp.update([empty_example, example])
    mixed_preds = list(torch_ner.pipe(docs))
    print(mixed_preds)
    assert len(mixed_preds) == 2

    assert len(mixed_preds[0]) == len(empty_doc)
    assert len(mixed_preds[1]) == len(doc)