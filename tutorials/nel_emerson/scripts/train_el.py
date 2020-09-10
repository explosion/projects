import typer
import random
import json
import pickle
from collections import Counter
from pathlib import Path

import spacy
from spacy.kb import KnowledgeBase
from spacy.util import minibatch, compounding


def main(json_loc: Path, kb_dir: Path, nlp_dir: Path, output_nlp_dir: Path, output_test_set: Path):
    """ Step 2: Once we have done the manual annotations, use it to train a new Entity Linking component. """
    nlp = spacy.load(nlp_dir)
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.load_bulk(kb_dir)

    dataset = []
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
        indices = [i for i, j in enumerate(gold_ids) if j == QID]
        train_dataset.extend(dataset[index] for index in indices[0:8])  # first 8 in training
        test_dataset.extend(dataset[index] for index in indices[8:10])  # last 2 in test

    # avoid artificial signals by reshuffling the datasets
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    TRAIN_DOCS = []
    for text, annotation in train_dataset:
        doc = nlp(text)  # to make this more efficient, you can use nlp.pipe() just once for all the texts
        TRAIN_DOCS.append((doc, annotation))

    entity_linker = nlp.create_pipe("entity_linker", config={"incl_prior": False, "kb": kb})
    nlp.add_pipe(entity_linker, last=True)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]
    print("Training the entity linker")
    with nlp.disable_pipes(*other_pipes):  # train only the entity_linker
        optimizer = nlp.begin_training()
        for itn in range(500):  # 500 iterations takes about a minute to train on this small dataset
            random.shuffle(TRAIN_DOCS)
            batches = minibatch(TRAIN_DOCS, size=compounding(4.0, 32.0, 1.001))  # increasing batch size
            losses = {}
            for batch in batches:
                nlp.update(
                    batch,
                    drop=0.2,  # prevent overfitting
                    losses=losses,
                    sgd=optimizer,
                )
            if itn % 50 == 0:
                print(itn, "Losses", losses)  # print the training loss
    print(itn, "Losses", losses)
    print()

    nlp.to_disk(output_nlp_dir)

    with open(output_test_set, "wb") as f:
        pickle.dump(test_dataset, f)


if __name__ == "__main__":
    typer.run(main)
