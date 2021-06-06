from typing import Iterable, Iterator
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from spacy import util

from thinc.api import Config

from scripts.azure.text_analytics import TextAnalyticsClient, ResponseDocument, Entity


DEFAULT_CONFIG = """
[azure_ner]
text_analytics_key = null
text_analytics_base_url = https://westus2.api.cognitive.microsoft.com/
text_analytics_endpoint = pii
text_analytics_domain = phi
extension_attr = "azure_ents"
use_extension_attr = true
"""

DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG)


@Language.factory("azure_ner", default_config=DEFAULT_CONFIG["azure_ner"])
def make_azure_entity_recognizer(
    nlp: Language,
    name: str,
    text_analytics_key: str,
    text_analytics_base_url: str,
    text_analytics_endpoint: str,
    text_analytics_domain: str,
    extension_attr: str,
    use_extension_attr: bool = True
):

    client = TextAnalyticsClient(
        text_analytics_key,
        text_analytics_base_url,
        text_analytics_endpoint,
        text_analytics_domain,
        default_language=nlp.lang,
    )
    return AzureEntityRecognizer(
        client,
        extension_attr,
        use_extension_attr,
        name,
    )


class AzureEntityRecognizer(Pipe):
    """Recognize entities using the Azure Text Analytics Entity Recognition API.

    Requires an active Azure Subscription with a Cognitive Services account.
    Follow the Prerequisite steps here: https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/quickstarts/client-libraries-rest-api?tabs=version-3-1&pivots=programming-language-csharp

    NOTE: This component is not really designed to be saved/loaded to/from disk. The main 
    reason for this is we'd have to save a private API key from Azure to disk which is a 
    security risk.
    """    
    def __init__(
        self,
        client: TextAnalyticsClient,
        extension_attr: str = "azure_ents",
        use_extension_attr: bool = True,
        name: str = "azure_ner",
    ):
        """Initialize AzureEntityRecognizer
        nlp (Language): spaCy Language Object
        client (TextAnalyticsClient): Text Analytics client
        extension_attr (str): Attribute name for the extension property to set the ents to
        use_extension_attr (bool): Use extension attr or set to doc.ents directly if False
        name (str): The component instance name.
        """
        self.name = name
        self.client = client
        self.use_extension_attr = use_extension_attr
        if self.use_extension_attr:
            self.extension_attr = extension_attr
            Doc.set_extension(self.extension_attr, default=[])

    def __call__(self, doc: Doc) -> Doc:
        """Extract entities from a single document using Azure Text Analytics API
        doc (Doc): Input doc
        RETURNS (Doc): Doc with extracted entities set
        """
        preds = self.predict([doc])
        self.set_annotations([doc], preds)
        return doc

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Extract entities from a stream of documents using Azure Text Analytics API.

        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.
        """
        for docs in util.minibatch(stream, size=batch_size):
            preds = self.predict(docs)
            self.set_annotations(docs, preds)
            yield from docs

    def predict(self, docs: Iterable[Doc]) -> Iterable[ResponseDocument]:
        """Extract entities from a batch of documents using Azure Text Analytics API.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS: Raw predictions from Azure Text Analytics API
        """
        res = self.client.predict([d.text for d in docs])
        return res.documents

    def set_annotations(self, docs: Iterable[Doc], preds: Iterable[ResponseDocument]):
        """Modify a batch of documents with predictions from the Azure Text Analytics API

        docs (Iterable[Doc]): The documents to modify.
        preds (Iterable[ResponseDocument]): Predictions from Azure
        """
        for doc, pred in zip(docs, preds):
            spacy_ents = []
            for entity in pred.entities:
                start = entity.offset
                end = entity.offset + entity.length
                label = entity.category
                spacy_ents.append(doc.char_span(start, end, label=label))

            if self.use_extension_attr:
                setattr(doc._, self.extension_attr, spacy_ents)
            else:
                # Note this does not currently preserve entities previously set in the pipeline.
                # So you would want to run this before an existing ner pipeline component.
                # e.g. nlp.add_pipe("azure_ner", before="ner")
                doc.ents = spacy_ents

            return doc
