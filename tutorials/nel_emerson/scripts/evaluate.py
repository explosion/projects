import typer
import pickle
from pathlib import Path

import spacy


def main(nlp_dir: Path, output_test_set: Path):
    """ Step 4: Evaluate the new Entity Linking component by applying it to unseen text. """
    nlp = spacy.load(nlp_dir)
    with open(output_test_set, "rb") as f:
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
        doc = nlp(text)   # to make this more efficient, you can use nlp.pipe() just once for all the texts
        for ent in doc.ents:
            if ent.text == "Emerson":
                print(f"Prediction: {ent.text}, {ent.label_}, {ent.kb_id_}")
        print()


if __name__ == "__main__":
    typer.run(main)
