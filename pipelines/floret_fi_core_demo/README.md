<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Demo floret vectors for Finnish

Train floret vectors on OSCAR and compare standard vectors vs. floret vectors on UD Finnish TDT and turku-ner-corpus.

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
| `tokenize-oscar` | Download, tokenize, and sentencize data |
| `train-fasttext-standard-vectors` | Train standard fasttext vectors |
| `train-floret-vectors` | Train floret vectors |
| `init-standard-unpruned-vectors` | Create a standard unpruned vectors model |
| `init-standard-vectors` | Create a standard vectors model |
| `init-floret-vectors` | Create a floret vectors model |
| `convert` | Convert the data to spaCy's format |
| `train-no-vectors` | Train the model without vectors |
| `train-standard-unpruned` | Train the model with standard, unpruned vectors |
| `train-standard` | Train the model with standard, pruned vectors |
| `train-floret` | Train the model with floret vectors |
| `evaluate` | Evaluate the models and export metrics |
| `convert-ner` | Convert the data to spaCy's format |
| `train-no-vectors-ner` | Train the model without vectors |
| `train-standard-unpruned-ner` | Train the model with standard, unpruned vectors |
| `train-standard-ner` | Train the model with standard, pruned vectors |
| `train-floret-ner` | Train the model with floret vectors |
| `evaluate-ner` | Evaluate the models and export metrics |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `tokenize-oscar` &rarr; `train-fasttext-standard-vectors` &rarr; `train-floret-vectors` &rarr; `init-standard-unpruned-vectors` &rarr; `init-standard-vectors` &rarr; `init-floret-vectors` &rarr; `convert` &rarr; `train-no-vectors` &rarr; `train-standard-unpruned` &rarr; `train-standard` &rarr; `train-floret` &rarr; `evaluate` &rarr; `convert-ner` &rarr; `train-no-vectors-ner` &rarr; `train-standard-unpruned-ner` &rarr; `train-standard-ner` &rarr; `train-floret-ner` &rarr; `evaluate-ner` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/UD_Finnish-TDT` | Git |  |
| `assets/turku-ner-corpus` | Git |  |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
