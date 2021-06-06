"""
Custom Entity Recognition pipeline component using Azure Text Analytics. 

This implementation is based on the Presidio example here: https://microsoft.github.io/presidio/samples/python/text_analytics/example_text_analytics_recognizer/
Needs setting up a Text Analytics resource as a prerequisite:
https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/cognitive-services/text-analytics/includes/create-text-analytics-resource.md
"""

from enum import Enum
from typing import Dict, Iterable, List, Optional
from pydantic import BaseModel
import requests


class RequestDocument(BaseModel):
    id: str
    text: str
    language: str


class RequestBody(BaseModel):
    documents: List[RequestDocument]


class Entity(BaseModel):
    offset: int
    length: int
    category: str
    subcategory: Optional[str]
    confidenceScore: Optional[float]


class ResponseDocument(BaseModel):
    id: str
    entities: List[Entity]


class ResponseBody(BaseModel):
    documents: List[ResponseDocument]


class Endpoint(str, Enum):
    GENERAL = "general"
    PII = "pii"


class TextAnalyticsClient:
    """Client for Azure Text Analytics Entity Recognition API."""

    def __init__(
        self,
        key: str,
        base_url: str,
        endpoint: Endpoint = Endpoint.PII,
        domain: str = "phi",
        default_language: str = "en",
    ):
        """Initialize TextAnalyticsClient
        key (str): The key used to authenticate to Text Analytics Azure Instance.
        base_url (str): Supported Cognitive Services or Text Analytics
            resource endpoints (protocol and hostname).
        endpoint (Endpoint): Endpoint for prediction. Defaults to PII.
        domain (str): Domain to use for recognition. Defaults to PHI.
        """
        self.__key = key
        self.base_url = base_url
        self.endpoint = endpoint
        self.domain = domain
        self.default_language = default_language

    def predict(
        self, texts: Iterable[str], language: Optional[str] = None
    ) -> ResponseBody:
        """Extract Azure entities from batch of texts
        texts (Iterable[str]): Input texts
        language (Optional[str]): Input text language.
        RETURNS (List[Dict]): List of recognized entities with character offsets
        """
        if not language:
            language = self.default_language

        if not texts:
            return ResponseBody(documents=[])

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": self.__key,
        }
        data = {
            "documents": [
                {"id": i, "language": language, "text": text}
                for i, text in enumerate(texts)
            ]
        }
        recognition_path = f"/text/analytics/v3.1-preview.5/entities/recognition/{self.endpoint}?domain={self.domain}"
        res = requests.post(
            self.base_url + recognition_path, json=data, headers=headers
        )
        data = res.json()
        response = ResponseBody(**data)
        return response
