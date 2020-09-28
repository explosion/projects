from typing import List, Callable, Tuple
import spacy
from transformers import AutoTokenizer, AutoModel
import stanza

from download_models import SPACY_MODELS, STANZA_MODELS, HF_TRF_MODELS


def run_spacy_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    """Run a pretrained spaCy pipeline"""
    if gpu:
        spacy.require_gpu(0)

    nlp = spacy.load(name)

    def run(texts: List[str]):
        nlp.pipe(texts)

    return run


def run_transformer_model(name: str) -> Callable[[List[str]], None]:
    """Run bare transformer model, outputting raw hidden-states"""
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    transformer = AutoModel.from_pretrained(name)

    def run(texts: List[str]):
        batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        transformer(**batch)

    return run


def run_stanza(name: str, gpu: bool) -> Callable[[List[str]], None]:
    """Run Stanza pretrained model"""
    lang = name.split("_")[0]
    package = name.split("_")[1]
    nlp = stanza.Pipeline(lang, package=package, use_gpu=gpu, verbose=False)

    def run(texts: List[str]):
        for text in texts:
            nlp(text)

    return run


def get_all_nlp_functions() -> List[Tuple[str, bool, Callable[[List[str]], None]]]:
    functions = []

    # spaCy functions
    for name in SPACY_MODELS:
        functions.append((f"spacy_{name}", True, run_spacy_model(name, gpu=True)))
        functions.append((f"spacy_{name}", False, run_spacy_model(name, gpu=False)))

    # HF transformers
    for name in HF_TRF_MODELS:
        functions.append((f"hf_trf_{name}", True, run_transformer_model(name)))

    # Stanza models
    for name in STANZA_MODELS:
        functions.append((f"stanza_{name}", False, run_stanza(name, gpu=False)))
        functions.append((f"stanza_{name}", True, run_stanza(name, gpu=True)))

    # TODO Flair
    # TODO UDPipe

    return functions
