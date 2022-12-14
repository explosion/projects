<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Categorization of emotions in Reddit posts (Text Classification)

This project uses spaCy to train a text classifier on the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) with options for a pipeline with and without transformer weights. To use the BERT-based config, change the `config` variable in the `project.yml`. 

> The goal of this project is to show how to train a spaCy classifier based on a csv file, not to showcase a model that's ready for production. The GoEmotions dataset has known flaws described [here](https://github.com/google-research/google-research/tree/master/goemotions#disclaimer) as well as label errors resulting from [annotator disagreement](https://www.youtube.com/watch?v=khZ5-AN-n2Y).


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
| `init-vectors` | Download vectors and convert to model |
| `preprocess` | Convert the corpus to spaCy's format |
| `train` | Train a spaCy pipeline using the specified corpus and config |
| `evaluate` | Evaluate on the test data and save the metrics |
| `package` | Package the trained model so it can be installed |
| `visualize` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` &rarr; `package` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/categories.txt` | URL | The categories to train |
| `assets/train.tsv` | URL | The training data |
| `assets/dev.tsv` | URL | The development data |
| `assets/test.tsv` | URL | The test data |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
