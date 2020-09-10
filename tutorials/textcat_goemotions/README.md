<!-- SPACY PROJECT: IGNORE -->

## Text categorization of emotions in Reddit posts

I'll be using this project as an internal reference so we can keep a number of
workflows straight. We can turn this into expanded user-facing docs later, but
for now we just need something we can all refer back to.

In order to avoid repeating lines, examples may tell you to start from one of
the following steps.

```bash

a. spacy project clone tutorials/textcat_goemotions myproject
b. cd myproject
c. spacy project assets
d. spacy project run preprocess
e. spacy init config --lang en  --pipeline textcat > configs/my_new_config.cfg
f. Edit project.yml, changing the CONFIG variable to my_new_config
g. spacy project run train
h. spacy project run evaluate
```

Note that step e isn't something we'll have exactly in the real workflow, it's
there as a helper for the examples.

## Basic spacy project workflow

```bash

spacy project clone tutorials/textcat_goemotions myproject
cd myproject
spacy project assets
spacy project run all
```

## Transformers

Install the `spacy-transformers` package:

```
pip install spacy-transformers
```

Edit `project.yml` to use the `configs/bert.cfg` and choose your GPU:

```
variables:
  NAME: "en_textcat_reddit"
  CONFIG: "bert"
  VERSION: "0.0.1"
  GPU_ID: 0
```

Use via project:

```
spacy project run all
```

Or directly:

```
spacy train configs/bert.cfg
```

## Tuning a hyper-parameter in the config

From step h), you can make a branch for your experiment, commit your config and
your metrics, and make a commit with your result:

```bash
git add configs/my_new_config.cfg metrics/my_new_config.json
git commit -m "Initial experiment. Scores TODO%."
```

Now you can change some parameter in your config, and retrain. For instance, you
could edit the `components.textcat.model.tok2vec.encode.width` value, changing
it to 32:

```yaml
[components.textcat.model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v1"
width = 32
depth = 4
window_size = 1
maxout_pieces = 3
```

Now you can retrain and reevaluate, and commit the updated config and metrics:

```
spacy project run train
spacy project run evaluate
git commit configs/my_new_config.cfg metrics/my_new_config.cfg -m "Scores TODO%"
```

## Adjusting a hyper-parameter on the command line

You can run experiments in a more lightweight way by running `spacy train`
directly and changing hyper-parameters on the command line.

From step h:

```bash
spacy train \
    configs/my_new_config.cfg \
    --components.textcat.model.tok2vec.encode.width 32
```

## Adding components from another model

Let's say you want to take tagger and NER components from the `en_core_web_sm`
model, and add a new textcat model that you'll train, while keeping the existing
models from the tagger and NER. This requires three changes to the config.

1. Add the components to the `nlp.pipeline`

```
[nlp]
pipeline = ["tagger", "ner", "textcat"]
```

2. Add the "sourced" components in the `[components]` block

This tells the config to build the NER and tagger components from the
`en_core_web_sm` config and to load their models from disk.

```
[components]
tagger = {"source": "en_core_web_sm"}
ner = {"source": "en_core_web_sm"}
```

3. Specify that the tagger and NER are "frozen"

This stops the weights of these models from being reset, and stops the
components from being updated.

```
[training]
frozen_components = ["tagger", "ner"]
```

## Training from a stream instead of a file

Create a Python file `plugins/my_stream.py` with the registered function to
stream from:

```python
from spacy.utils import registry
from spacy.language import Language
from spacy.gold import Example
from functools import partial


@registry.readers("my_data_stream.v1")
def configure_training_stream(some_arg: Path, another_arg: float):
    return partial(
        stream_training_examples,
        some_arg=some_arg,
        another_arg=another_arg
    )


def stream_training_examples(nlp: Language, some_arg, another_arg) -> Iterable[Example]:
    # TODO
    for example in todo(nlp, ...):
        yield example
```

Then change your config to use your registered function:

```
[training.train_corpus]
@readers = "my_data_stream.v1"
some_arg = "my_train_path/"
another_arg = 42.6

[training.dev_corpus]
@readers = "my_data_stream.v1"
some_arg = "my_dev_path/"
another_arg = 89.1
```

And add the `-c` argument to your `spacy train` calls:

```
spacy train configs/my_config.cfg -c plugins/my_loader.py
```

## Using embeddings from a spaCy package

```bash
spacy train \
    configs/cnn.cfg \
    --training.vectors "en_vectors_web_lg" \
    --components.textcat.model.tok2vec.embed.also_use_static_vectors true
```

## Making and using new embeddings

Add the asset in your `project.yml`:

```yaml
assets:
  - dest: 'assets/vectors.zip'
    url: 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
```

Then download the asset and run the `init-vectors` step:

```
spacy project assets
spacy project run init-vectors
```

Use the vectors:

```yaml
spacy train \ configs/cnn.cfg \ --training.vectors "assets/en_fasttext_vectors"
\ --components.textcat.model.tok2vec.embed.also_use_static_vectors true
```

## Developing a custom PyTorch model defined in an imported file

In a new file `plugins/pytorch_bilstm.py`:

```python
from spacy.util import registry
from thinc.api import PyTorchWrapper
import json
import torch.nn


@registry.architectures("mine.PyTorchBiLSTMTextcat.v1")
def make_pytorch_lstm_textcat(
    vocab_path: Path
    n_labels: int,
    lstm_width: int,
    embed_width: int,
    depth: int,
    dropout: float
) -> Model[List[Doc], Floats2d]:
    """Thinc wrapper around a PyTorch lstm textcat model. The model takes a
    list of Doc objects, converts them into a list of strings, and uses PyTorch
    to encode the strings into integers, embed, and run the LSTM model.
    """
    # TODO: Ugh, this won't actually work -- the path won't be available in
    # deserialization.
    with open(vocab_path) as file_:
        vocab = json.load(file_)
    torch_model = _PyTorchLSTMTextcat(
                vocab,
                n_labels,
                lstm_width,
                embed_width,
                depth,
                dropout
            )

    model = chain(
        Model("docs2strings", docs2strings),
        PyTorchWrapper(torch_model)
    )
    return model


def docs2text(model, docs, is_train):
    return [[token.text for token in doc] for doc in docs], lambda d_docs: []


class _PyTorchLSTMTextcat(torch.nn.Module):
    def __init__(
        self,
        vocab: Dict[str, int],
        n_labels: int,
        lstm_width: int,
        embed_width:int,
        depth: int,
        dropout: float
    ):
        self.vocab = vocab
        self.embed = torch.nn.Embedding(embed_width, len(self.vocab) + 1)
        self.rnn = torch.nn.LSTM(
            input_size=embed_width,
            hidden_size=lstm_width,
            num_layers=config.n_layers, dropout=dropout,
            bidirectional=False
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(lstm_width, n_labels)
        self.softmax = torch.nn.Softmax()
        self.all = torch.nn.Sequential(
            self.embed,
            self.rnn,
            self.dropout,
            self.linear,
            self.softmax
        )

    def forward(self, inputs: List[str]) -> torch.Tensor:
        return self.all(self.encode_integers(inputs))

    def encode_integers(self, sbatch: List[str]) -> torch.Tensor:
        oov = len(self.vocab)
        ibatch = [[self.vocab.get(word, oov) for word in words] for words in sbatch]
        tbatch = _pad_torch_tensors(ibatch)
        return tbatch

def _pad_torch_tensors(ibatch):
    raise NotImplementedError("TODO")
```

Now we can make a new config, changing the model block:

```bash
cp configs/cnn.cfg configs/torch_lstm.cfg
```

And inside our config we replace the `components.textcat.model` block with:

```yaml
[components.textcat.model]
@architectures = "mine.PyTorchBiLSTMTextcat.v1"
vocab_path = "assets/vocab.json"
n_labels = 28
lstm_width = 128
embed_width = 128,
depth = 4
dropout = 0.2
```

We then run our model with:

```bash
spacy train \
    corpus/train.spacy \
    corpus/dev.spacy \
    configs/torch_lstm.cfg \
    -c plugins/pytoch_lstm.py
```

## Pretraining

Add the following to the project.yml:

```yaml
commands:
  - name: pretrain
    help: 'Run language model pretraining to warm-start the tok2vec'
    script:
      - 'mkdir -p pretrain/'
      - 'python -m spacy pretrain assets/reddit-100k.jsonl pretrain/
        configs/cnn.cfg'
    deps:
      - 'assets/reddit-100k.jsonl'
    outputs:
      - 'pretrain/model100.bin'
```

The weights are then passed into the `training.init_tok2vec` argument:

```bash
spacy train \
    corpus/train.spacy \
    corpus/dev.spacy \
    configs/cnn.cfg \
    --training.init_tok2vec pretrain/model100.bin
```

This is one case where the outputs are a little awkward because we don't really
know which checkpoint will be the actual artefact. I suppose the pretrain script
needs to output a `model-best.bin`.

It will be fairly standard to include pretrained weights as an asset, rather
than asking the user to always retrain them.

## Parallel training with Ray (TODO)

TODO

## Remote storage with DVC (TODO)
