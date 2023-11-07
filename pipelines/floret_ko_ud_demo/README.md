<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Demo floret vectors for UD Korean Kaist

Train floret vectors on OSCAR and compare no vectors, standard vectors, and floret vectors on UD Korean Kaist.

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
| `train-no-vectors-model` | Train the model without vectors |
| `train-standard-unpruned-model` | Train the model with standard, unpruned vectors |
| `train-standard-model` | Train the model with standard, pruned vectors |
| `train-floret-model` | Train the model with floret vectors |
| `evaluate` | Evaluate the models and export metrics |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `tokenize-oscar` &rarr; `train-fasttext-standard-vectors` &rarr; `train-floret-vectors` &rarr; `init-standard-unpruned-vectors` &rarr; `init-standard-vectors` &rarr; `init-floret-vectors` &rarr; `convert` &rarr; `train-no-vectors-model` &rarr; `train-standard-unpruned-model` &rarr; `train-standard-model` &rarr; `train-floret-model` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/UD_Korean-Kaist` | Git |  |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
