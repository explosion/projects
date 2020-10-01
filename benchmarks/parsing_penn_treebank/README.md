<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Dependency Parsing (Penn Treebank)

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

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
| `init-config` | Generate default config |
| `corpus` | Convert the data to spaCy's format |
| `vectors` | Convert, truncate and prune the vectors. |
| `train` | Train the full pipeline |
| `evaluate` | Evaluate on the test data and save the metrics |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `vectors` &rarr; `corpus` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://nightly.spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/PTB_SD_3_3_0/train.gold.conll` | Local | Training data (not available publicly so you have to add the file yourself) |
| `assets/PTB_SD_3_3_0/dev.gold.conll` | Local | Development data (not available publicly so you have to add the file yourself) |
| `assets/PTB_SD_3_3_0/test.gold.conll` | Local | Test data (not available publicly so you have to add the file yourself) |
| `assets/vectors.zip` | URL | GloVe vectors |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
