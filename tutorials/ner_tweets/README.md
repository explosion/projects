<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Detecting people entities in tweets (Named Entity Recognition)

This project demonstrates how to improve spaCy's pretrained models by
augmenting the training data and adapting it to a different domain.

**Weak supervision** is the practice of using imprecise annotators to label a
dataset. These annotators take in the form of *labelling functions*, and may
contain a single heuristic, a rule, or a model trained from related data.

Once these functions are defined, we annotate the dataset and fit a Hidden Markov
Model (HMM) to "pool" our annotators together. The result is a single, unified
model that contains the pooled knowledge of each annotator. We can then use
this unified model to re-annotate our entire training dataset.

In this example, we will be using `skweak` as our weak supervision framework.
It contains primitives that allows us to define our own labelling functions. 
Throughout this project, we will be using the following annotators:
- **A model-based annotator based on `en_core_web_lg`: we can use existing
    spaCy models to bootstrap our annotations.
- **A model-based annotator based on a model trained on the Broad Twitter
    Corpus**: this gives us an edge due to having similar domains.
- **A gazetteer-based annotator from Crunchbase**: gazetteer can be thought
    simply as a list of entities. In this case, we obtain all person names from
    Crunchbase to create an annotator of business personalities.
- **A heuristic for finding proper names**: a simple implementation that checks
    for proper names in a given tweet. If there are two proper names joined by
    a prefix ("**van** Gogh", "**del** Pilar", "**de** Castro", etc.), then
    it's most likely a person's name. 
- **A heuristic for finding full names based on a list**: we obtained a list of
    first names from skweak's data utilities, and use that to condition our
    search. If a full name is found, and a proper name comes after it, then
    it's most likely a person's name.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `preprocess` | Convert raw inputs into spaCy's preferred formats |
| `decompress` | Decompress relevant assets that will be used latter by our weak supervision model |
| `augment` | Augment an input dataset via weak supervision then split it into training and evaluation datasets |
| `train` | Train a named entity recognition model |
| `train-with-augmenter` | Train a named entity recognition model with a spaCy built-in augmenter |
| `evaluate` | Evaluate various experiments and export the computed metrics |
| `package` | Package the trained model so it can be installed |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `decompress` &rarr; `augment` &rarr; `train` &rarr; `evaluate` |
| `setup` | `preprocess` &rarr; `decompress` |
| `finetune` | `augment` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/train_initial.jsonl`](assets/train_initial.jsonl) | Local | The training dataset to augment. Later we'll divide this further to create a dev set |
| [`assets/test.jsonl`](assets/test.jsonl) | Local | The held-out test dataset to evaluate our output against. |
| `assets/btc.tar.gz` | URL | A model trained on the Broad Twitter Corpus to help in annotation (63.7 MB) |
| `assets/crunchbase.json.gz` | URL | A list of crunchbase entities to aid the gazetteer annotator (8.56 MB) |
| `assets/wikidata_tokenised.json.gz` | URL | A list of wikipedia entities to aid the gazetteer annotator (21.1 MB) |
| `assets/first_names.json` | URL | A list of first names to help our heuristic annotator |
| `assets/en_orth_variants.json` | URL | Orth variants to use for data augmentation |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
