import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin

from rel_pipe import make_relation_extractor  # make the factory work
from rel_model import (
    create_relation_model,
    create_candidates,
    create_layer,
)  # make the config work
from spacy.training.example import Example


def main(trained_pipeline: Path, test_data: Path):
    nlp = spacy.load(trained_pipeline)

    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(nlp.vocab)
    examples = []
    for gold in docs:
        text = gold.text
        pred = nlp.make_doc(text)
        pred.ents = gold.ents
        for name, proc in nlp.pipeline:
            pred = proc(pred)
        print()
        print(f"Text: {text}")
        print(f"spans: {[(e.start, e.text, e.label_) for e in pred.ents]}")
        for value, rel_dict in pred._.rel.items():
            gold_labels = [k for (k, v) in gold._.rel[value].items() if v == 1.0]
            # only printing cases where there is a gold label
            if gold_labels:
                print(f" pair: {value} --> gold labels: {gold_labels} --> predicted values: {rel_dict}")
        print()
        examples.append(Example(pred, gold))

    results = nlp.get_pipe("relation_extractor").score(examples)
    print(results)


if __name__ == "__main__":
    typer.run(main)
