<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Demo NER in a new pipeline (Named Entity Recognition)

A minimal demo NER project for spaCy v3 adapted from the spaCy v2 [`train_ner.py`](https://github.com/explosion/spaCy/blob/v2.3.x/examples/training/train_ner.py) example script for creating an NER component in a new pipeline.

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
| `download` | Download a spaCy model with pretrained vectors |
| `convert` | Convert the data to spaCy's binary format |
| `create-config` | Create a new config with an NER pipeline component |
| `train` | Train the NER model |
| `train-with-vectors` | Train the NER model with vectors |
| `evaluate` | Evaluate the model and export metrics |
| `package` | Package the trained model as a pip package |
| `visualize-model` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `convert` &rarr; `create-config` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/train.json`](assets/train.json) | Local | Demo training data converted from the v2 `train_ner.py` example with `srsly.write_json("train.json", TRAIN_DATA)` |
| [`assets/dev.json`](assets/dev.json) | Local | Demo development data |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
