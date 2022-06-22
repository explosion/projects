<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: NEL Benchmark

Pipeline for benchmarking NEL approaches (incl. candidate generation and entity disambiguation).

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
| `setup` | Install dependencies |
| `preprocess` | Preprocess test datasets |
| `download` | Download a model with pretrained vectors and NER component |
| `create_kb` | Create the knowledge base and write it to file |
| `compile_corpora` | Compile corpora, separated in in train/dev/test sets |
| `train` | Train a new Entity Linking component. Pass --gpu_id GPU_ID to train with GPU |
| `evaluate` | Evaluation on the test set |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `setup` &rarr; `preprocess` &rarr; `download` &rarr; `create_kb` &rarr; `compile_corpora` &rarr; `train` &rarr; `evaluate` |
| `training` | `create_kb` &rarr; `compile_corpora` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/reddit.zip` | URL | Entity linking dataset scraped from Reddit. See [paper](https://arxiv.org/abs/2101.01228). |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
