import spacy 
from spacy.language import Language
from spacy.tokens import Doc

@Language.factory("transfer-ent.v1", requires=["doc._.ents"])
def make_transfer_component(nlp: Language, name: str, span_key: str):
    return TransferEntComponent(nlp, name, span_key)

class TransferEntComponent:
    """Transfer doc.ents to doc.spans"""

    def __init__(self, nlp: Language, name: str, span_key: str):
        self.nlp = nlp
        self.name = name
        self.span_key = span_key

    def __call__(self, doc:Doc) -> Doc:
        doc.spans[self.span_key] = list(doc.ents)
        doc.set_ents([])
        return doc