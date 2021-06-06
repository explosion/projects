from typing import List, Optional, Union, Iterable, Dict, Any
import murmurhash
import spacy
from spacy.language import Language
from spacy.training import Example
import copy

from prodigy.recipes.compare import get_questions as get_compare_questions
from prodigy.recipes.compare import get_printer as get_compare_printer
from prodigy.models.ner import EntityRecognizer, ensure_sentencizer
from prodigy.models.matcher import PatternMatcher
from prodigy.components.db import connect
from prodigy.components.preprocess import split_sentences, add_tokens, make_raw_doc
from prodigy.components.sorters import prefer_uncertain
from prodigy.components.loaders import get_stream
from prodigy.core import recipe
from prodigy.util import (
    combine_models,
    set_hashes,
    log,
    split_string,
    get_labels,
    copy_nlp,
)
from prodigy.util import INPUT_HASH_ATTR, TASK_HASH_ATTR, msg

from scripts.custom_functions import *
from scripts.azure.azure_ner_pipe import make_azure_entity_recognizer


@recipe(
    "ner.correct.anonymous",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy model with an entity recognizer", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    text_analytics_key=("Azure Text Analytics API Key", "option", "key", str),
    text_analytics_base_url=("Azure Text Analytics API Key", "option", "url", str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    unsegmented=("Don't split sentences", "flag", "U", bool),
    # fmt: on
)
def make_gold(
    dataset: str,
    spacy_model: str,
    source: Union[str, Iterable[dict]],
    text_analytics_key: str,
    text_analytics_base_url: str,
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    update: bool = False,
    exclude: Optional[List[str]] = None,
    unsegmented: bool = False,
) -> Dict[str, Any]:
    """
    Create gold data for NER by correcting a model's suggestions.
    """
    log("RECIPE: Starting recipe ner.correct", locals())
    nlp = spacy.load(spacy_model)
    labels = label  # comma-separated list or path to text file
    model_labels = nlp.pipe_labels.get("ner", [])
    if not labels:
        labels = model_labels
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")
    # Check if we're annotating all labels present in the model or a subset
    no_missing = len(set(labels).intersection(set(model_labels))) == len(model_labels)
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    stream = get_stream(
        source, loader=loader, rehash=True, dedup=True, input_key="text"
    )
    if not unsegmented:
        stream = split_sentences(nlp, stream)

    def add_anonymized_tokens(nlp: Language, stream: Iterable[dict]) -> Iterable[dict]:
        """Add tokens with anonymized PII Entities.
        Based on the implementation of the built-in Prodigy add_tokens method
        """
        stream = list(stream)
        texts = (eg["text"] for eg in stream)
        for eg, doc in zip(stream, nlp.pipe(texts)):
            task = copy.deepcopy(eg)
            pii_ents_map = {t: s for s in doc._.azure_ents for t in s}

            tokens = []
            for t in doc:
                if t in pii_ents_map:
                    text = t.shape_
                else:
                    text = t.text

                tokens.append(
                    {
                        "text": text,
                        "start": t.idx,
                        "end": t.idx + len(t),
                        "id": t.i,
                        "ws": bool(t.whitespace_),
                    }
                )

            task["tokens"] = tokens
            task["original_text"] = task["text"]

            anonymized_text = ""
            for t in tokens:
                anonymized_text += t["text"]
                if t["ws"]:
                    anonymized_text += " "

            task["text"] = anonymized_text
            yield task

    def make_tasks(nlp: Language, stream: Iterable[dict]) -> Iterable[dict]:
        """Add a 'spans' key to each example, with predicted entities."""
        texts = ((eg["text"], eg) for eg in stream)
        for doc, eg in nlp.pipe(texts, as_tuples=True, batch_size=10):
            task = copy.deepcopy(eg)
            spans = []
            for ent in doc.ents:
                if labels and ent.label_ not in labels:
                    continue
                spans.append(
                    {
                        "token_start": ent.start,
                        "token_end": ent.end - 1,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "text": ent.text,
                        "label": ent.label_,
                        "source": spacy_model,
                        "input_hash": eg[INPUT_HASH_ATTR],
                    }
                )
            task["spans"] = spans
            task = set_hashes(task)
            yield task

    def make_update(answers: Iterable[dict]) -> None:
        log(f"RECIPE: Updating model with {len(answers)} answers")
        examples = []
        for eg in answers:
            if eg["answer"] == "accept":
                # Use original text in the model training
                doc = nlp.make_doc(eg["original_text"])
                annots = [
                    (span["start"], span["end"], span["label"])
                    for span in eg.get("spans", [])
                ]
                examples.append(Example.from_dict(doc, {"entities": annots}))
        nlp.update(examples)

    # Add AzureEntityRecognizer pipeline for PII entities
    azure_ner_config = {"text_analytics_key": text_analytics_key}
    if text_analytics_base_url:
        azure_ner_config["text_analytics_base_url"] = text_analytics_base_url
    nlp.add_pipe("azure_ner", config=azure_ner_config)

    stream = add_anonymized_tokens(nlp, stream)
    stream = make_tasks(nlp, stream)

    return {
        "view_id": "ner_manual",
        "dataset": dataset,
        "stream": stream,
        "update": make_update if update else None,
        "exclude": exclude,
        "config": {
            "lang": nlp.lang,
            "labels": labels,
            "exclude_by": "input",
            "force_stream_order": True,
        },
    }
