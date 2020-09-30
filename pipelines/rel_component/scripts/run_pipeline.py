import typer
from spacy.lang.en import English
from pathlib import Path

from spacy.training.example import Example
from thinc.api import Config

from rel_component import make_relation_extractor  # make the factory work
from rel_model import create_relation_model, create_candidates, create_layer  # make the config work

from spacy.tokens import DocBin


default_model_config = """
[model]
@architectures = "my_rel_model.v1"
nO = 5

[model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v1"
pretrained_vectors = null
width = 5
depth = 2
embed_size = 300
window_size = 1
maxout_pieces = 3
subword_features = true

[model.get_candidates]
@architectures = "my_rel_candidate_generator.v1"

[model.output_layer]
@architectures = "my_rel_output_layer.v1"
nI = null
nO = null
"""


def main(data_file: Path):
    nlp = English()

    # set up dummy "NER"
    patterns = [
        {"label": "CITY", "pattern": [{"LOWER": "new"}, {"LOWER": "york"}]},
        {"label": "CITY", "pattern": [{"LOWER": "london"}]},
        {"label": "COUNTRY", "pattern": [{"LOWER": "united"}, {"LOWER": "states"}]},
        {"label": "COUNTRY", "pattern": [{"LOWER": "united"}, {"LOWER": "kingdom"}]},
    ]
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)

    # set up the Relation Extraction component
    nlp.add_pipe(
        "relation_extractor",
        config=Config().from_str(default_model_config),
    )

    # read example data
    train_examples = []
    doc_bin = DocBin(store_user_data=True).from_disk(data_file)
    docs = doc_bin.get_docs(nlp.vocab)
    for doc in docs:
        example = Example(nlp.make_doc(doc.text), doc)
        train_examples.append(example)

    nlp.begin_training(lambda: train_examples)

    text = "London is the capital of the united kingdom, just like the capital of the united states is new york."
    doc = nlp(text)
    ents = doc.ents
    print([(e.text, e.label_) for e in ents])
    print()
    print("rel", doc._.rel)


if __name__ == "__main__":
    typer.run(main)
