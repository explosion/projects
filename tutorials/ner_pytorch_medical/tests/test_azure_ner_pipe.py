import pytest
from typing import List
import spacy
from spacy.tokens import Doc
from scripts.azure.azure_ner_pipe import (
    make_azure_entity_recognizer,
    AzureEntityRecognizer,
)
from scripts.azure.text_analytics import (
    TextAnalyticsClient,
    ResponseBody,
    ResponseDocument,
)


class MockTextAnalyticsClient(TextAnalyticsClient):
    def __init__(self, entities):
        self.entities = entities

    def predict(self, texts: List[str], language: str = "en") -> ResponseBody:
        res_doc = {"id": "a1", "entities": self.entities}

        return ResponseBody(documents=[ResponseDocument(**res_doc)])


TEXT = "Name: Kabir Khan \nSSN: 444-34-1394"
ENTITIES = [
    {"text": "Kabir Khan", "offset": 6, "length": 10, "category": "Person"},
    {
        "text": "444-34-1394",
        "offset": 23,
        "length": 11,
        "category": "USSocialSecurityNumber",
    },
]


@pytest.fixture()
def mock_client():
    return MockTextAnalyticsClient(ENTITIES)


@pytest.fixture()
def nlp():
    return spacy.blank("en")


def test_predict(mock_client, nlp):
    doc = nlp.make_doc(TEXT)
    rec = AzureEntityRecognizer(mock_client)
    doc = rec(doc)
    assert isinstance(doc, Doc)

    spacy_ents = [
        doc.char_span(6, 16, label="Person"),
        doc.char_span(23, 34, label="USSocialSecurityNumber"),
    ]
    assert len(doc.ents) == 0
    assert getattr(doc._, rec.extension_attr) == spacy_ents


def test_extension_attr(mock_client, nlp):
    doc = nlp.make_doc(TEXT)

    rec = AzureEntityRecognizer(mock_client, extension_attr="other_attr")
    doc = rec(doc)
    assert isinstance(doc, Doc)
    assert hasattr(doc._, "other_attr")

    assert len(doc.ents) == 0
    assert len(getattr(doc._, rec.extension_attr)) == 2

    rec2 = AzureEntityRecognizer(mock_client, use_extension_attr=False)
    doc2 = nlp.make_doc(TEXT)
    doc2 = rec2(doc2)
    assert len(doc2.ents) == 2
