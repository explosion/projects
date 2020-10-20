import typer
import csv
import os
from pathlib import Path

import spacy
from spacy.kb import KnowledgeBase


def main(entities_loc: Path, vectors_model: str, kb_loc: Path, nlp_dir: Path):
    """ Step 1: create the Knowledge Base in spaCy and write it to file """

    # First: create a simpel model from predefined vectors and a simpel EntityRuler component
    # A more realistic use-case would use a pretrained NER model instead
    nlp = spacy.load(vectors_model, exclude="parser, tagger, ner")
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [{"label": "PERSON", "pattern": [{"LOWER": "emerson"}]}]
    ruler.add_patterns(patterns)
    nlp.add_pipe("sentencizer")

    name_dict, desc_dict = _load_entities(entities_loc)

    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        # Set arbitrary value for frequency
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)

    for qid, name in name_dict.items():
        # set 100% prior probability P(entity|alias) for each unique name
        kb.add_alias(alias=name, entities=[qid], probabilities=[1])

    qids = name_dict.keys()
    probs = [0.3 for qid in qids]
    # ensure that sum([probs]) <= 1 when setting aliases
    kb.add_alias(alias="Emerson", entities=qids, probabilities=probs)  #

    print(f"Entities in the KB: {kb.get_entity_strings()}")
    print(f"Aliases in the KB: {kb.get_alias_strings()}")
    print()
    kb.to_disk(kb_loc)
    if not os.path.exists(nlp_dir):
        os.mkdir(nlp_dir)
    nlp.to_disk(nlp_dir)


def _load_entities(entities_loc: Path):
    """ Helper function to read in the pre-defined entities we want to disambiguate to. """
    names = dict()
    descriptions = dict()
    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]
            names[qid] = name
            descriptions[qid] = desc
    return names, descriptions


if __name__ == "__main__":
    typer.run(main)
