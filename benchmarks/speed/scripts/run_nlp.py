from typing import Callable, List

import torch
import typer
import timeit
from pathlib import Path

import spacy
import stanza
from transformers import AutoTokenizer, AutoModel
import flair
from flair.models import MultiTagger
from flair.tokenization import SegtokSentenceSplitter

from data_reader import read_data
from logger import create_logger


def main(txt_dir: Path, result_dir: Path ,model_name: str, gpu: bool):
    log_run = create_logger(result_dir)
    data = read_data(txt_dir)
    articles = len(data)
    chars = sum([len(d) for d in data])
    words = sum([len(d.split()) for d in data])

    nlp_function = _get_run(model_name, gpu)
    start = timeit.default_timer()
    nlp_function(data)
    end = timeit.default_timer()

    s = end - start
    log_run(
        name=model_name,
        gpu=gpu,
        articles=articles,
        characters=chars,
        words=words,
        seconds=s,
    )


def _get_run(model_name: str, gpu: bool) -> Callable[[List[str]], None]:
    if model_name.startswith("spacy_"):
        spacy_name = model_name.replace("spacy_", "")
        return _run_spacy_model(spacy_name, gpu)

    if model_name.startswith("stanza_"):
        stanza_name = model_name.replace("stanza_", "")
        return _run_stanza_model(stanza_name, gpu)

    if model_name.startswith("hf_trf_"):
        hf_trf_name = model_name.replace("hf_trf_", "")
        return _run_transformer_model(hf_trf_name)

    if model_name.startswith("flair_"):
        flair_name = model_name.replace("flair_", "")
        return _run_flair_model(flair_name, gpu)

    # TODO UDPipe

    print(f"Can not parse model name {model_name}, prefix should be one of: [spacy_, stanza_, hf_trf_]")


def _run_spacy_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    """Run a pretrained spaCy pipeline"""
    if gpu:
        spacy.require_gpu(0)
    nlp = spacy.load(name)

    def run(texts: List[str]):
        nlp.pipe(texts)

    return run


def _run_transformer_model(name: str) -> Callable[[List[str]], None]:
    """Run bare transformer model, outputting raw hidden-states"""
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    transformer = AutoModel.from_pretrained(name)

    def run(texts: List[str]):
        batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        transformer(**batch)

    return run


def _run_stanza_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    """Run a Stanza pretrained model"""
    lang = name.split("_")[0]
    package = name.split("_")[1]
    nlp = stanza.Pipeline(lang, package=package, use_gpu=gpu, verbose=False)

    def run(texts: List[str]):
        # No multi-document option available in Stanza?
        for text in texts:
            nlp(text)

    return run


def _run_flair_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    """Run a pretrained Flair pipeline"""
    annot_list = name.split("_")
    if gpu:
        tagger = MultiTagger.load(annot_list)
    else:
        flair.device = torch.device('cpu')
        tagger = MultiTagger.load(annot_list)

    def run(texts: List[str]):
        # TODO: multi-document option?
        for text in texts:
            splitter = SegtokSentenceSplitter()
            sentences = splitter.split(text)
            tagger.predict(sentences, verbose=False)

    return run


if __name__ == "__main__":
    typer.run(main)
