<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Healthsea-Spancat

This spaCy project uses the Healthsea dataset to compare the results between the Spancat and NER architecture

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
| `preprocess` | Format .jsonl annotations into .spaCy training format for NER and Spancat |
| `train_ner` | Train an NER model |
| `train_spancat` | Train a Spancat model |
| `evaluate_ner` | Evaluate the trained NER model |
| `evaluate_spancat` | Evaluate the trained Spancat model |
| `evaluate` | Evaluate NER vs Spancat on the dev dataset |
| `reset` | Reset the project to its original state and delete all training process |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train_ner` &rarr; `train_spancat` &rarr; `evaluate` |
| `ner` | `preprocess` &rarr; `train_ner` &rarr; `evaluate_ner` |
| `spancat` | `preprocess` &rarr; `train_spancat` &rarr; `evaluate_spancat` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/annotation.jsonl` | URL | NER annotations exported from Prodigy with 5000 examples and 2 labels |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->