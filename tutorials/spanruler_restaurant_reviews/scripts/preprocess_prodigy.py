import copy
from pathlib import Path
from typing import Dict, Union

import spacy
import srsly
import typer
from spacy.tokens import DocBin

from prodigy.components.preprocess import add_tokens
from prodigy.util import set_hashes
from scripts.rules import restaurant_span_rules

Arg = typer.Argument
Opt = typer.Option


def get_text_annotations(
    input_file: Path = Arg(..., help="Input path for the raw spacy files."),
):
    """
    Turn the raw spaCy files generated from IOB data into dictionaries with
    text and spans.

    Returns a dictionary with text, spans, and annotator ID and a dictionary with
    text.
    """
    # open IOB data
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(input_file)
    docs = list(doc_bin.get_docs(nlp.vocab))

    org_annotations = []
    texts = []

    for doc in docs:
        text = doc.text
        spans = [
            {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
            for ent in doc.ents
        ]

        # append data to lists
        org_annotations.append(
            {
                "text": text,
                "spans": spans,
                "_annotator_id": "original_annotations",
                "_session_id": "original_annotations",
            }
        )
        texts.append({"text": text})

    return org_annotations, texts


def get_model_data(
    texts: Dict[str, any] = Arg(..., help="Dictionary of texts in the dataset."),
    model: Union[str, Path] = Arg(..., help="The trained NER model."),
):
    """
    Create JSON data with model annotations from the trained NER model.

    Returns a dictionary with text, spans, and annotator ID.
    """
    # load trained model
    nlp = spacy.load(model)

    texts_copy = copy.deepcopy(texts)

    for line in texts_copy:
        text = line["text"]
        doc = nlp(text)
        spans = [
            {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
            for ent in doc.ents
        ]

        line["spans"] = spans
        line["_annotator_id"] = "ner_model"
        line["_session_id"] = "ner_model"

    return texts_copy


def get_ruler_data(
    texts: Dict[str, any] = Arg(..., help="Dictionary of texts in the dataset."),
):
    """
    Create JSON data with annotations from the SpanRuler patterns.

    Returns a dictionary with text, spans, and annotator ID.
    """
    nlp = spacy.blank("en")

    # add span ruler pattern pipe on blank tokenizer
    patterns = restaurant_span_rules()
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(patterns)

    texts_copy = copy.deepcopy(texts)

    for line in texts_copy:
        text = line["text"]
        doc = nlp(text)
        spans = [
            {"start": span.start_char, "end": span.end_char, "label": span.label_}
            for span in doc.spans["ruler"]
        ]

        line["spans"] = spans
        line["_annotator_id"] = "ruler"
        line["_session_id"] = "ruler"

    return texts_copy


def preprocess_prodigy(
    input_file: Path = Arg(..., help="Input path for the raw IOB files."),
    output_file: Path = Arg(..., help="Output path for the processed jsonl files."),
    model: Path = Arg(..., help="The trained NER model."),
    include_ruler: bool = Opt(
        False, help="Whether to include the ruler in the outputted annotations."
    ),
):
    """
    Preprocess the raw IOB files from MIT Restaurant Reviews into JSONL with
    the different annotations (original, model, ruler) as multiple annotators.

    Outputs a JSONL file with annotations from the original dataset, the trained
    NER model, and the SpanRuler patterns (optional).
    """
    # obtain original and ner annotations and combine
    org_annotations, texts = get_text_annotations(input_file)
    model_annotations = get_model_data(texts, model)
    combined_annotations = org_annotations + model_annotations

    # generate ruler data and combine with annotations if True
    if include_ruler:
        ruler_annotations = get_ruler_data(texts)
        combined_annotations += ruler_annotations

    # add tokens to stream
    nlp = spacy.blank("en")
    stream = add_tokens(nlp=nlp, stream=combined_annotations, skip=True)

    # set hashes in stream only on text key so they're the same across examples
    stream = (
        set_hashes(
            eg, task_keys=("text"), ignore=("spans", "_annotator_id", "_session_id")
        )
        for eg in stream
    )

    # write to file
    srsly.write_jsonl(output_file, stream)


if __name__ == "__main__":
    typer.run(preprocess_prodigy)
