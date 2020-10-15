from typing import Callable, List

import re
import torch
import typer
import timeit
from pathlib import Path
import logging
from wasabi import msg

from data_reader import read_data, rebatch_texts
from logger import create_logger
from spacy.util import minibatch


def main(
    txt_dir: Path,
    result_dir: Path,
    library,
    name: str,
    gpu: bool,
    batch_size: int = 256,
    n_texts: int=0
):
    data = read_data(txt_dir, limit=n_texts)
    articles = len(data)
    if articles == 0:
        msg.fail(
            f"Could not read any data from {txt_dir}: make sure a corpus of .txt files is available."
        )
    chars = sum([len(d) for d in data])
    words = sum([len(d.split()) for d in data])

    nlp_function = _get_run(library, name, gpu)
    start = timeit.default_timer()
    nlp_function(data, batch_size)
    end = timeit.default_timer()

    log_run = create_logger(result_dir)
    s = end - start
    log_run(
        library=library,
        name=name,
        gpu=gpu,
        articles=articles,
        characters=chars,
        words=words,
        seconds=s,
    )


def _get_run(library: str, name: str, gpu: bool) -> Callable[[List[str]], None]:
    if library == "spacy":
        return _run_spacy_model(name, gpu)

    if library == "stanza":
        return _run_stanza_model(name, gpu)

    if library == "hf_trf":
        return _run_transformer_model(name, gpu)

    if library == "flair":
        return _run_flair_model(name, gpu)

    if library == "ud_pipe":
        return _run_ud_pipe(name)

    msg.fail(
        f"Can not parse models for library {library}. "
        f"Known libraries are: ['spacy', 'stanza', 'hf_trf', 'flair', 'ud_pipe']",
        exits=1,
    )


def _run_spacy_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    """Run a pretrained spaCy pipeline"""
    import spacy

    if gpu:
        spacy.require_gpu(0)
    nlp = spacy.load(name)

    def run(texts: List[str], batch_size: int):
        list(nlp.pipe(texts, batch_size=batch_size))

    return run


def _run_transformer_model(name: str, gpu) -> Callable[[List[str]], None]:
    """Run bare transformer model, outputting raw hidden-states"""
    from transformers import AutoTokenizer, AutoModel
    import torch
    if gpu:
        torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    transformer = AutoModel.from_pretrained(name)
    if gpu:
        transformer = transformer.cuda()

    def run(texts: List[str], batch_size: int):
        transformer.eval()
        for batch in minibatch(texts, batch_size // 20):
            batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            if gpu:
                batch["input_ids"] = batch["input_ids"].to("cuda:0")
                batch["attention_mask"] = batch["attention_mask"].to("cuda:0")
            transformer(**batch)

    return run


def _run_stanza_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    """Run a Stanza pretrained model"""
    import stanza

    lang = name.split("_")[0]
    package = name.split("_")[1]
    nlp = stanza.Pipeline(lang, package=package, use_gpu=gpu, verbose=False)

    def run(texts: List[str], batch_size: int):
        # No batch parsing option available in Stanza I think? instead we have to
        # re-batch, concatenating with \n\n
        for text in rebatch_texts(texts, batch_size):
            nlp(text)

    return run


def _run_flair_model(name: str, gpu: bool) -> Callable[[List[str]], None]:
    """Run a pretrained Flair pipeline"""
    import flair
    from flair.models import MultiTagger
    from flair.tokenization import SegtokSentenceSplitter

    logging.getLogger("flair").setLevel(logging.ERROR)
    annot_list = name.split("_")
    if not gpu:
        flair.device = torch.device("cpu")
    tagger = MultiTagger.load(annot_list)
    splitter = SegtokSentenceSplitter()

    def run(texts: List[str], batch_size: int):
        # No batch parsing option available in Flair I think? instead we have to
        # re-batch, concatenating with \n\n
        for text in rebatch_texts(texts, batch_size):
            sentences = splitter.split(text)
            tagger.predict(sentences, verbose=False)

    return run


def _run_ud_pipe(name: str):
    from ufal.udpipe import Model, Sentence

    model = Model.load(name)
    tokenizer = model.newTokenizer(model.DEFAULT)

    def run(texts: List[str], batch_size: int):
        # TODO: multi-document option?
        for text in rebatch_texts(texts, batch_size):
            tokenizer.setText(text)
            sentences = []
            sentence = Sentence()
            while tokenizer.nextSentence(sentence):
                sentences.append(sentence)
                sentence = Sentence()
            for s in sentences:
                model.tag(s, model.DEFAULT)
                model.parse(s, model.DEFAULT)

    return run


if __name__ == "__main__":
    typer.run(main)
