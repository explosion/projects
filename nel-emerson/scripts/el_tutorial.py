import spacy
from spacy.kb import KnowledgeBase
from spacy.util import minibatch, compounding

import os
import csv
import json
import random
import pickle
from pathlib import Path
from collections import Counter

input_dir = Path.cwd().parent / "input"
output_dir = Path.cwd().parent / "output"
prodigy_dir = Path.cwd().parent / "prodigy"


def load_entities():
    """ Helper function to read in the pre-defined entities we want to disambiguate to. """
    entities_loc = input_dir / "entities.csv"

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


def create_kb():
    """ Step 1: create the Knowledge Base in spaCy and write it to file """
    nlp = spacy.load("en_core_web_lg")
    name_dict, desc_dict = load_entities()

    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)  # 342 is an arbitrary value here

    for qid, name in name_dict.items():
        kb.add_alias(alias=name, entities=[qid], probabilities=[1])  # 100% prior probability P(entity|alias)

    qids = name_dict.keys()
    probs = [0.3 for qid in qids]
    kb.add_alias(alias="Emerson", entities=qids, probabilities=probs)  # sum([probs]) should be <= 1 !

    print(f"Entities in the KB: {kb.get_entity_strings()}")
    print(f"Aliases in the KB: {kb.get_alias_strings()}")
    print()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    kb.dump(output_dir / "my_kb")
    nlp.to_disk(output_dir / "my_nlp")


def train_el():
    """ Step 2: Once we have done the manual annotations, use it to train a new Entity Linking component. """
    nlp = spacy.load(output_dir / "my_nlp")
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.load_bulk(output_dir / "my_kb")

    dataset = []
    json_loc = prodigy_dir / "emerson_annotated_text.jsonl"
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            text = example["text"]
            if example["answer"] == "accept":
                QID = example["accept"][0]
                offset = (example["spans"][0]["start"], example["spans"][0]["end"])
                links_dict = {QID: 1.0}
            dataset.append((text, {"links": {offset: links_dict}}))

    gold_ids = []
    for text, annot in dataset:
        for span, links_dict in annot["links"].items():
            for link, value in links_dict.items():
                if value:
                    gold_ids.append(link)
    print("Statistics of manually annotated data:")
    print(Counter(gold_ids))
    print()

    train_dataset = []
    test_dataset = []
    for QID in ['Q312545', 'Q48226', 'Q215952']:
        indices = [i for i,j in enumerate(gold_ids) if j == QID]
        train_dataset.extend(dataset[index] for index in indices[0:8])  # first 8 in training
        test_dataset.extend(dataset[index] for index in indices[8:10])  # last 2 in test

    # avoid artificial signals by reshuffling the datasets
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    TRAIN_DOCS = []
    for text, annotation in train_dataset:
        doc = nlp(text)
        TRAIN_DOCS.append((doc, annotation))

    entity_linker = nlp.create_pipe("entity_linker", config={"incl_prior": False})
    entity_linker.set_kb(kb)
    nlp.add_pipe(entity_linker, last=True)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]
    print("Training the entity linker")
    with nlp.disable_pipes(*other_pipes):   # train only the entity_linker
        optimizer = nlp.begin_training()
        for itn in range(500):   # 500 iterations takes about a minute to train on this small dataset
            random.shuffle(TRAIN_DOCS)
            batches = minibatch(TRAIN_DOCS, size=compounding(4.0, 32.0, 1.001))   # increasing batch size
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=0.2,   # prevent overfitting
                    losses=losses,
                    sgd=optimizer,
                )
            if itn % 50 == 0:
                print(itn, "Losses", losses)   # print the training loss
    print(itn, "Losses", losses)
    print()

    nlp.to_disk(output_dir / "my_nlp_el")

    with open(output_dir / "test_set.pkl", "wb") as f:
        pickle.dump(test_dataset, f)


def eval():
    """ Step 3: Evaluate the new Entity Linking component by applying it to unseen text. """
    nlp = spacy.load(output_dir / "my_nlp_el")
    with open(output_dir / "test_set.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    text = "Tennis champion Emerson was expected to win Wimbledon."
    doc = nlp(text)
    print(text)
    for ent in doc.ents:
        print(ent.text, ent.label_, ent.kb_id_)
    print()

    for text, true_annot in test_dataset:
        print(text)
        print(f"Gold annotation: {true_annot}")
        doc = nlp(text)
        for ent in doc.ents:
            if ent.text == "Emerson":
                print(f"Prediction: {ent.text}, {ent.label_}, {ent.kb_id_}")
        print()


if __name__ == "__main__":
    create_kb()
    # after creating the KB, it can be used to run the Prodigy recipe and create the manual annotations

    train_el()
    eval()
