<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Demo floret vectors

Train floret vectors and load them into a spaCy vectors model.

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
| `tokenize-oscar` | Download, tokenize, and sentencize data |
| `train-floret` | Train floret vectors |
| `init-floret-vectors` | Create a floret vectors model |
| `floret-nn` | Demo nearest neighbors for intentional OOV misspelling 'outdooor' |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `tokenize-oscar` &rarr; `train-floret` &rarr; `init-floret-vectors` &rarr; `floret-nn` |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
