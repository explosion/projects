import spacy
from spacy.tokens import Doc
from spacy.training.example import Example
from scripts.custom_functions import make_torch_entity_recognizer


def test_torch_ner_init_update_predict():
    nlp = spacy.blank("en")
    torch_ner = nlp.add_pipe("torch_ner")

    torch_ner.add_label("B")
    torch_ner.add_label("I")
    torch_ner.add_label("O")

    empty_doc = Doc(nlp.vocab)
    doc = nlp.make_doc("Test doc.")

    empty_doc_preds = torch_ner.predict(empty_doc)
    assert empty_doc_preds == []

    empty_preds = torch_ner.predict([])
    assert empty_preds == []

    docs = [empty_doc, doc]

    empty_example = Example(empty_doc, empty_doc)
    example = Example(doc, doc)

    nlp.initialize()
    nlp.update([empty_example, example])
    mixed_preds = torch_ner.predict(docs)
    print(mixed_preds)
    assert len(mixed_preds) == 2

    assert len(mixed_preds[0]) == len(empty_doc)
    assert len(mixed_preds[1]) == len(doc)