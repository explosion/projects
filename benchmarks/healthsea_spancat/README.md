<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Healthsea-Spancat

This spaCy project uses the Healthsea dataset to compare the performance between the Spancat and NER architecture.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[Weasel documentation](https://github.com/explosion/weasel).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `preprocess` | Format .jsonl annotations into .spacy training format for NER and Spancat |
| `train_ner` | Train an NER model |
| `train_spancat` | Train a Spancat model |
| `evaluate_ner` | Evaluate the trained NER model |
| `evaluate_spancat` | Evaluate the trained Spancat model |
| `evaluate` | Evaluate NER vs Spancat on the dev dataset and create a detailed performance analysis which is saved in the metrics folder |
| `reset` | Reset the project to its original state and delete all training process |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train_ner` &rarr; `train_spancat` &rarr; `evaluate` |
| `ner` | `preprocess` &rarr; `train_ner` &rarr; `evaluate_ner` |
| `spancat` | `preprocess` &rarr; `train_spancat` &rarr; `evaluate_spancat` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/annotation.jsonl` | URL | NER annotations exported from Prodigy with 5000 examples and 2 labels |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->