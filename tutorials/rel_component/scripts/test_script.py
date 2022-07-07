import random
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example

# make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


def main(trained_pipeline: Path, input_str: str):
    nlp = spacy.load(trained_pipeline)
    print("Pipe Names:", nlp.pipe_names)
    print("Pipeline Components:", nlp.pipeline)
    print("Input String:", input_str)

    print('--------NLP Tokens--------')
    doc = nlp(input_str)
    for tok in doc:
        print(tok.text)

    print('--------NLP Entities--------')
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    print('--------RELS--------')
    for tok in doc:
        for child in tok.children:
            print('child:', child, 'type:', type(child))

if __name__ == "__main__":
    typer.run(main)
