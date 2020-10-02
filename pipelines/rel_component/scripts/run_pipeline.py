import typer
from spacy.lang.en import English
from pathlib import Path

from spacy.training.example import Example
from thinc.api import Config

from rel_pipe import make_relation_extractor  # make the factory work
from rel_model import create_relation_model, create_candidates, create_layer  # make the config work

from spacy.tokens import DocBin


default_model_config = """
[model]
@architectures = "rel_model.v1"
nO = null

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
@architectures = "rel_cand_generator.v1"
max_length = 6

[model.create_candidate_tensor]
@architectures = "rel_cand_tensor.v1"

[model.output_layer]
@architectures = "rel_output_layer.v1"
"""

bert_model_config = """
[model]
@architectures = "rel_model.v1"
nO = null

[model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0

[model.tok2vec.pooling]
@layers = "reduce_mean.v1"

[model.get_candidates]
@architectures = "rel_cand_generator.v1"

[model.create_candidate_tensor]
@architectures = "rel_cand_tensor.v1"

[model.output_layer]
@architectures = "rel_output_layer.v1"
"""


def main(data_file: Path):
    import numpy
    numpy.random.seed(342)
    nlp = English()

    # set up dummy "NER"
    patterns = [
        {"label": "LOC", "pattern": [{"LOWER": "new"}, {"LOWER": "york"}]},
        {"label": "LOC", "pattern": [{"LOWER": "london"}]},
        {"label": "LOC", "pattern": [{"LOWER": "united"}, {"LOWER": "states"}]},
        {"label": "LOC", "pattern": [{"LOWER": "united"}, {"LOWER": "kingdom"}]},
        {"label": "LOC", "pattern": [{"LOWER": "amsterdam"}]},
        {"label": "LOC", "pattern": [{"LOWER": "netherlands"}]},
    ]
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)

    # nlp.add_pipe("transformer")

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

    optimizer = nlp.begin_training(lambda: train_examples)

    print()
    print("PREDICTING (0)")
    print()
    _evaluate(nlp)
    print()

    print("TRAINING")
    print()

    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
        if i % 50 == 0:
            print(i, losses)

    print()
    print("PREDICTING (50)")
    print()
    _evaluate(nlp)
    print()


def _evaluate(nlp):
    # text = "London is the capital of the United Kingdom, just like the capital of the United States is New York."
    text = "Amsterdam is the capital of the Netherlands."
    doc = nlp(text)
    ents = doc.ents
    print()
    print("spans", [(e.start, e.text, e.label_) for e in ents])
    print()
    for value, rel_dict in doc._.rel.items():
        print(f"rel for {value}: {rel_dict}")


if __name__ == "__main__":
    typer.run(main)