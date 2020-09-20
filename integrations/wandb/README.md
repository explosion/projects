<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Weights & Biases integration

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

Use [Weights & Biases](https://www.wandb.com/) for logging of training experiments. This project template uses the IMDB Movie Review Dataset and includes two workflows: `log` for training a simple text classification model and logging the results to Weights & Biases (works out-of-the-box and only requires the `[training.logger]` to be set in the config) and `parameter-search` for programmatically creating variants of the config for a simple hyperparameter grid search, running the experiments and logging the results.

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
| `install` | Install dependencies and log in to Weights & Biases |
| `data` | Extract the gold-standard annotations |
| `train` | Train a model using the default config |
| `configs-search` | Create variations of the initial, default file for IMDB sentiment classification using different combinations of hyperparameters |
| `train-search` | Run customized training runs for hyperparameter search using the created configs |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `log` | `data` &rarr; `train` |
| `parameter-search` | `data` &rarr; `configs-search` &rarr; `train-search` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://nightly.spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/aclImdb_v1.tar.gz` | URL | Movie Review Dataset for sentiment analysis by Maas et al., ACL 2011. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->