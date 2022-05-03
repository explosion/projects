<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Training a spaCy Coref Model

This project trains a coreference model for spaCy using OntoNotes.


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
| `prep-data` | Rehydrate the data using OntoNotes |
| `preprocess` | Convert the data to spaCy's format |
| `train-cluster` | Train the clustering component |
| `prep-span-data` | Prepare data for the span predictor component. |
| `train-span-predictor` | Train the span predictor component. |
| `assemble` | Assemble parts into complete coref pipeline. |
| `eval` | Evaluate model on the test set. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `prep` | `prep-data` &rarr; `preprocess` |
| `train` | `train-cluster` &rarr; `prep-span-data` &rarr; `train-span-predictor` &rarr; `assemble` |
| `all` | `prep-data` &rarr; `preprocess` &rarr; `train-cluster` &rarr; `prep-span-data` &rarr; `train-span-predictor` &rarr; `assemble` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/` | Git | CoNLL-2012 scripts and dehydrated data. |
| `/home/USER/ontonotes5/data` | Local | Ensure you have a local copy of OntoNotes: https://catalog.ldc.upenn.edu/LDC2013T19 |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
