<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Dependency Parsing (Penn Treebank)

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
| `corpus` | Convert the data to spaCy's format |
| `vectors` | Convert, truncate and prune the vectors. |
| `train` | Train the full pipeline |
| `evaluate` | Evaluate on the test data and save the metrics |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `vectors` &rarr; `corpus` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/PTB_SD_3_3_0/train.gold.conll` | Local | Training data (not available publicly so you have to add the file yourself) |
| `assets/PTB_SD_3_3_0/dev.gold.conll` | Local | Development data (not available publicly so you have to add the file yourself) |
| `assets/PTB_SD_3_3_0/test.gold.conll` | Local | Test data (not available publicly so you have to add the file yourself) |
| `assets/vectors.zip` | URL | GloVe vectors |
| `assets/orth_variants.json` | URL | A file containing orth variants for data augmentation |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
