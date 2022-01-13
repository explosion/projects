import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens.span_group import SpanGroup


@Language.factory("transfer-ent.v1", requires=["doc._.ents"])
def make_transfer_component(nlp: Language, name: str, span_key: str):
    return TransferEntComponent(nlp, name, span_key)


class TransferEntComponent:
    """Transfer doc.ents to doc.spans"""

    def __init__(self, nlp: Language, name: str, span_key: str):
        self.nlp = nlp
        self.name = name
        self.span_key = span_key

    def __call__(self, doc: Doc) -> Doc:
        if self.span_key not in doc.spans:
            doc.spans[self.span_key] = SpanGroup(doc)
        doc.spans[self.span_key].extend(list(doc.ents))
        doc.set_ents([])
        return doc
