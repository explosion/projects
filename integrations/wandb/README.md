<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Weights & Biases integration with W&B Sweeps

Use [Weights & Biases](https://www.wandb.com/) for logging of training experiments. This project template uses the IMDB Movie Review Dataset and includes two workflows: `log` for training a simple text classification model and logging the results to Weights & Biases (works out-of-the-box and only requires the `[training.logger]` to be set in the config) and `parameter-search` for running a hyperparameter search using [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps), running the experiments and logging the results.

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
| `install` | Install dependencies and log in to Weights & Biases |
| `data` | Extract the gold-standard annotations |
| `train` | Train a model using the default config |
| `train-search` | Run customized training runs for hyperparameter search using [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps) |
| `clean` | Remove intermediate files. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `log` | `data` &rarr; `train` |
| `parameter-search` | `data` &rarr; `train-search` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/aclImdb_v1.tar.gz` | URL | Movie Review Dataset for sentiment analysis by Maas et al., ACL 2011. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
