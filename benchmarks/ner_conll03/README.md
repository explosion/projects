<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Named Entity Recognition (CoNLL-2003)

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
| `install` | Install dependencies |
| `corpus` | Convert the data to spaCy's format |
| `vectors` | Convert, truncate and prune the vectors. |
| `train` | Train the full pipeline |
| `evaluate` | Evaluate on the test data and save the metrics |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `install` &rarr; `vectors` &rarr; `corpus` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/vectors.zip` | URL | GloVe vectors |
| `assets/conll2003/dev.iob` | Local | Development data (not available publicly so you have to add the file yourself) |
| `assets/conll2003/test.iob` | Local | Test data (not available publicly so you have to add the file yourself) |
| `assets/conll2003/train.iob` | Local | Training data (not available publicly so you have to add the file yourself) |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->