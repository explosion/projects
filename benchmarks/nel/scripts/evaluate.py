import typer
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

# we need to import this to parse the custom reader from the config
from custom_functions import create_docbin_reader


def main(nlp_dir: Path, dev_set: Path):
    """ Step 4: Evaluate the new Entity Linking component by applying it to unseen text. """
    nlp = spacy.util.load_model_from_path(nlp_dir)
    examples = []
    with open(dev_set, "rb") as f:
        doc_bin = DocBin().from_disk(dev_set)
        docs = doc_bin.get_docs(nlp.vocab)
        for doc in docs:
            examples.append(Example(nlp(doc.text), doc))

    print()
    print("RESULTS ON THE DEV SET:")
    for example in examples:
        print(example.text)
        print(f"Gold annotation: {example.reference.ents[0].kb_id_}")
        print(f"Predicted annotation: {example.predicted.ents[0].kb_id_}")
        print()

    print()
    print("RUNNING THE PIPELINE ON UNSEEN TEXT:")
    text = "Tennis champion Emerson was expected to win Wimbledon."
    doc = nlp(text)
    print(text)
    for ent in doc.ents:
        print(ent.text, ent.label_, ent.kb_id_)
    print()



if __name__ == "__main__":
    typer.run(main)
