<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Train floret vectors from Wikipedia and OSCAR

This project downloads, extracts and preprocesses texts from Wikipedia and
OSCAR and trains vectors with [floret](https://github.com/explosion/floret).

By default, the project trains floret vectors for Macedonian.


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
| `extract-wikipedia` | Convert Wikipedia XML to JSONL with wikiextractor |
| `tokenize-wikipedia` | Tokenize and sentencize Wikipedia |
| `tokenize-oscar` | Tokenize and sentencize OSCAR dataset |
| `create-input` | Concatenate tokenized input texts |
| `train-floret-vectors` | Train floret vectors |
| `train-fasttext-vectors` | Train fastText vectors |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `extract-wikipedia` &rarr; `tokenize-wikipedia` &rarr; `tokenize-oscar` &rarr; `create-input` &rarr; `train-floret-vectors` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `/scratch/vectors/downloaded/wikipedia/mkwiki-latest-pages-articles.xml.bz2` | URL |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
