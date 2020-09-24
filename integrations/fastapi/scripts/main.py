from typing import List, Dict, Any
from enum import Enum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
from spacy.tokens import Doc


class ModelName(str, Enum):
    # Enum of the available models. This allows the API to raise a more specific
    # error if an invalid model is provided.
    en_core_web_sm = "en_core_web_sm"
    en_core_web_md = "en_core_web_md"
    en_core_web_lg = "en_core_web_lg"
    en_core_web_trf = "en_core_web_trf"


DEFAULT_MODEL = ModelName.en_core_web_sm
MODEL_NAMES = [model.value for model in ModelName]
MODELS = {name: spacy.load(name) for name in MODEL_NAMES}
print(f"Loaded {len(MODEL_NAMES)} models: {MODEL_NAMES}")


class Article(BaseModel):
    # Schema for a single article in a batch of articles to process
    text: str


class RequestModel(BaseModel):
    articles: List[Article]
    model: ModelName = DEFAULT_MODEL


class ResponseModel(BaseModel):
    # This is the schema of the expected response and depends on what you
    # return from get_data.

    class Batch(BaseModel):
        class Entity(BaseModel):
            label: str
            start: int
            end: int

        text: str
        ents: List[Entity] = []

    result: List[Batch]


def get_data(doc: Doc) -> Dict[str, Any]:
    """Extract the data to return from the REST API given a Doc object. Modify
    this function to include other data."""
    ents = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]
    return {"text": doc.text, "ents": ents}


# Set up the FastAPI app and define the endpoints
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.get("/models", summary="List all loaded models")
def get_models() -> List[str]:
    """Return a list of all available loaded models."""
    return MODEL_NAMES


@app.post("/process/", summary="Process batches of text", response_model=ResponseModel)
def process_articles(query: RequestModel):
    """Process a batch of articles and return the entities predicted by the
    given model. Each record in the data should have a key "text".
    """
    nlp = MODELS[query.model]
    response_body = []
    texts = (article.text for article in query.articles)
    for doc in nlp.pipe(texts):
        response_body.append(get_data(doc))
    return {"result": response_body}
