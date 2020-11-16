<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Categorization of emotions in Reddit posts (Text Classification)

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

This project uses spaCy to train a text classifier on the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) with options for a pipeline with and without transformer weights. To use the BERT-based config, change the `config` variable in the `project.yml`.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://nightly.spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `init-vectors` | Download vectors and convert to model |
| `preprocess` | Convert the corpus to spaCy's format |
| `train` | Train a spaCy pipeline using the specified corpus and config |
| `evaluate` | Evaluate on the test data and save the metrics |
| `package` | Package the trained model so it can be installed |
| `visualize` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` &rarr; `package` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://nightly.spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/categories.txt` | URL | The categories to train |
| `assets/train.tsv` | URL | The training data |
| `assets/dev.tsv` | URL | The development data |
| `assets/test.tsv` | URL | The test data |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

## Usage

If you want to use the BERT-based config ([`bert.cfg`](configs/bert.cfg)), make
sure you have `spacy-transformers` installed:

```
pip install spacy-transformers
```

You can choose your GPU by setting the `gpu_id` variable in the
[`project.yml`](project.yml).

### Tuning a hyper-parameter in the config

To change hyperparameters, you can edit the [config](configs) (or create a new
custom config). For instance, you could edit the
`components.textcat.model.tok2vec.encode.width` value, changing it to `32`:

```ini
[components.textcat.model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v1"
width = 32
depth = 4
window_size = 1
maxout_pieces = 3
```

Now you can retrain and reevaluate, and commit the updated config and metrics:

```bash
spacy project run train
spacy project run evaluate
git commit configs/my_new_config.cfg metrics/my_new_config.cfg -m "Scores TODO%"
```

You can also run experiments in a more lightweight way by running `spacy train`
directly and
[overwriting](https://nightly.spacy.io/usage/training#config-overrides)
hyperparameters on the command line:

```bash
spacy train \
    configs/my_new_config.cfg \
    --components.textcat.model.tok2vec.encode.width 32
```

### Adding components from another model

Let's say you want to take tagger and NER components from the `en_core_web_sm`
model, and add a new textcat model that you'll train, while keeping the existing
models from the tagger and NER. This requires three changes to the config.

1. Add the components to the `nlp.pipeline`.

   ```ini
   [nlp]
   pipeline = ["tagger", "ner", "textcat"]
   ```

2. Add the "sourced" components in the `[components]` block. This tells the
   config to build the NER and tagger components from the `en_core_web_sm`
   config and to load their models from disk.

   ```ini
   [components]
   tagger = {"source": "en_core_web_sm"}
   ner = {"source": "en_core_web_sm"}
   ```

3. Specify that the tagger and NER are "frozen". This stops the weights of these
   models from being reset, and stops the components from being updated.

   ```ini
   [training]
   frozen_components = ["tagger", "ner"]
   ```

### Using embeddings from a spaCy package

```bash
spacy train \
    configs/cnn.cfg \
    --training.vectors "en_vectors_web_lg" \
    --components.textcat.model.tok2vec.embed.also_use_static_vectors true
```

### Making and using new embeddings

Uncomment the asset in your [`project.yml`](project.yml):

```yaml
assets:
  - dest: 'assets/vectors.zip'
    url: 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
```

Then download the asset and run the `init-vectors` command:

```bash
spacy project assets
spacy project run init-vectors
```

Use the vectors:

```bash
spacy train \
    configs/cnn.cfg \
    --training.vectors "assets/en_fasttext_vectors" \
    --components.textcat.model.tok2vec.embed.also_use_static_vectors true
```
