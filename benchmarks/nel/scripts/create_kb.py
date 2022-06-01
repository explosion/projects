"""
Functionality for creating the knowledge base from downloaded assets and by querying Wikipedia's API.
"""

import typer
import pickle
from typing import Dict, Union, Set

import os
from pathlib import Path

import spacy
from spacy.kb import KnowledgeBase

import reddit

ENTITY_DATA_TYPE = Dict[str, Dict[str, Union[Set[str], str, int]]]
# Max. allowed Wikipedia API batch size with descriptions.
MAX_WIKI_API_BATCH_SIZE = 20


def main(dataset_id: str, vectors_model: str, temp_dir: Path):
    """ Create the Knowledge Base in spaCy and write it to file.

     dataset_id (dataset_id): Dataset ID.
     vectors_model (str): Name of model with word vectors to use.
     temp_dir (Path): Path to save knowledge base and NLP pipeline at.
     """

    assert dataset_id in ("reddit",)
    dataset_path = Path(os.path.join("assets", dataset_id))

    # Create a simpel model from a model with an NER component
    nlp = spacy.load(vectors_model, exclude="parser, tagger, lemmatizer")
    nlp.add_pipe("sentencizer", first=True)

    # Load Wiki entities for Reddit dataset, fetch additional information such as their description.
    entity_info, annotations, failed_lookups = reddit.parse_external_corpus(dataset_path)

    print(f"Constructing knowledge base with {len(entity_info)} entries")
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)
    for qid, info in entity_info.items():
        desc_doc = nlp(info["description"])
        kb.add_entity(entity=qid, entity_vector=desc_doc.vector, freq=info["frequency"])
        for name in info["names"]:
            kb.add_alias(alias=name.replace("_", " "), entities=[qid], probabilities=[1])

    # Serialize knowledge base & entity information.
    entity_info_path = os.path.join("assets", dataset_id, "entities.pkl")
    failed_lookups_path = os.path.join("assets", dataset_id, "entities_failed_lookups.pkl")
    annotations_path = os.path.join("assets", dataset_id, "annotations.pkl")
    for to_serialize in (
        (entity_info_path, entity_info), (failed_lookups_path, failed_lookups), (annotations_path, annotations)
    ):
        with open(to_serialize[0], 'wb') as fp:
            pickle.dump(to_serialize[1], fp)
    kb.to_disk(os.path.join(temp_dir, f"{dataset_id}.kb"))
    nlp_dir = os.path.join(temp_dir, f"{dataset_id}.nlp")
    if not os.path.exists(nlp_dir):
        os.mkdir(nlp_dir)
    nlp.to_disk(nlp_dir)


if __name__ == "__main__":
    typer.run(main)
    # main(
    #     "reddit",
    #     "en_core_web_md",
    #     ""
    # )
