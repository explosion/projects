<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Multilabel Textcat rehearsal (Text Classification)

A minimal demo showcasing the experimental rehearsal feature on a textcat_multilabel component. This project trains a textcat_multilabel pipeline on cooking data, the trained model will then be rehearsed with data it hasn't seen before. The goal of the project is to showcase how rehearsal can compensate the catastrophoic forgetting problem when training on top of a pre-trained model.

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
| `convert` | Convert the data to spaCy's binary format. Split the data for training and rehearsal. |
| `train` | Train the textcat model |
| `update` | Train on top of the trained textcat model without rehearse |
| `rehearse` | Rehearse the trained textcat_multilabel model. |
| `evaluate` | Evaluate all the models and export metrics |
| `evaluate_textcat` | Evaluate the trained textcat_multilabel model and export metrics |
| `evaluate_update` | Evaluate the update textcat model and export metrics |
| `evaluate_rehearse` | Evaluate the rehearsed model and export metrics |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `convert` &rarr; `train` &rarr; `update` &rarr; `rehearse` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/cooking-train.jsonl`](assets/cooking-train.jsonl) | Local | Training data from cooking.stackexchange.com |
| [`assets/cooking-dev.jsonl`](assets/cooking-dev.jsonl) | Local | Development data from cooking.stackexchange.com |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
