"""
Custom Prodigy recipe to perform manual annotation of entity links,
given an existing NER model and a knowledge base performing candidate generation.
You can run this project without having Prodigy or using this recipe:
sample results are stored in assets/emerson_annotated_text.jsonl
"""

import spacy
from spacy.kb import KnowledgeBase

import prodigy
from prodigy.models.ner import EntityRecognizer
from prodigy.components.loaders import TXT
from prodigy.util import set_hashes
from prodigy.components.filters import filter_duplicates

import csv
from pathlib import Path


@prodigy.recipe(
    "entity_linker.manual",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .txt file", "positional", None, Path),
    nlp_dir=("Path to the NLP model with a pretrained NER component", "positional", None, Path),
    kb_loc=("Path to the KB", "positional", None, Path),
    entity_loc=("Path to the file with additional information about the entities", "positional", None, Path),
)
def entity_linker_manual(dataset, source, nlp_dir, kb_loc, entity_loc):
    # Load the NLP and KB objects from file
    nlp = spacy.load(nlp_dir)
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.load_bulk(kb_loc)
    model = EntityRecognizer(nlp)

    # Read the pre-defined CSV file into dictionaries mapping QIDs to the full names and descriptions
    id_dict = dict()
    with entity_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            id_dict[row[0]] = (row[1], row[2])

    # Initialize the Prodigy stream by running the NER model
    stream = TXT(source)
    stream = [set_hashes(eg) for eg in stream]
    stream = (eg for score, eg in model(stream))

    # For each NER mention, add the candidates from the KB to the annotation task
    stream = _add_options(stream, kb, id_dict)
    stream = filter_duplicates(stream, by_input=True, by_task=False)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "choice",
        "config": {"choice_auto_accept": True},
    }


def _add_options(stream, kb, id_dict):
    """ Define the options the annotator will be given, by consulting the candidates from the KB for each NER span. """
    for task in stream:
        text = task["text"]
        for span in task["spans"]:
            start_char = int(span["start"])
            end_char = int(span["end"])
            mention = text[start_char:end_char]

            candidates = kb.get_candidates(mention)
            if candidates:
                options = [{"id": c.entity_, "html": _print_url(c.entity_, id_dict)} for c in candidates]

                # we sort the options by ID
                options = sorted(options, key=lambda r: int(r["id"][1:]))

                # we add in a few additional options in case a correct ID can not be picked
                options.append({"id": "NIL_otherLink", "text": "Link not in options"})
                options.append({"id": "NIL_ambiguous", "text": "Need more context"})

                task["options"] = options
                yield task


def _print_url(entity_id, id_dict):
    """ For each candidate QID, create a link to the corresponding Wikidata page and print the description """
    url_prefix = "https://www.wikidata.org/wiki/"
    name, descr = id_dict.get(entity_id)
    option = "<a href='" + url_prefix + entity_id + "'>" + entity_id + "</a>: " + descr
    return option
