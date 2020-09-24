from typing import List, Callable, Tuple
import spacy
from spacy.lang.en import English
from transformers import AutoTokenizer, AutoModel


def spacy_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    if gpu:
        spacy.require_gpu(0)

    # nlp = spacy.load("en_core_web_md")    TODO once we have v3 models
    nlp = English()

    def run(texts: List[str]):
        nlp.pipe(texts)

    return run


def transformer_model(name: str) -> Callable[[List[str]], None]:
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    # transformer = AutoModel.from_pretrained(name)

    def run(texts: List[str]):
        batch = tokenizer(texts, padding=True, truncation=True)
        # transformer(batch)

    return run


def get_all_nlp_functions() -> List[Tuple[str, bool, Callable[[List[str]], None]]]:
    functions = []

    # spaCy functions
    functions.append(
        ("spacy_en_core_web_md", False, spacy_model("en_core_web_md", gpu=False))
    )
    functions.append(
        ("spacy_en_core_web_md", True, spacy_model("en_core_web_md", gpu=True))
    )
    functions.append(
        ("spacy_en_core_web_trf", False, spacy_model("en_core_web_trf", gpu=False))
    )
    functions.append(
        ("spacy_en_core_web_trf", True, spacy_model("en_core_web_trf", gpu=True))
    )

    # HF transformers
    functions.append(("hf_trf_roberta-base", True, transformer_model("roberta-base")))

    # TODO Stanza
    # TODO Flair

    return functions
