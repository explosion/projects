import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin

from rel_pipe import make_relation_extractor  # make the factory work
from rel_model import create_relation_model, create_candidates, create_layer  # make the config work


def main(trained_pipeline: Path, test_data: Path):
    nlp = spacy.load(trained_pipeline)

    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(nlp.vocab)
    for doc in docs:
        text = doc.text
        predicted_doc = nlp(text)
        print()
        ents = predicted_doc.ents
        print(f"Text: {text}")
        print(f"spans: {[(e.start, e.text, e.label_) for e in ents]}")
        for value, rel_dict in predicted_doc._.rel.items():
            print(f"rel for {value}: {rel_dict}")


if __name__ == "__main__":
    typer.run(main)
