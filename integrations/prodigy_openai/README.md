<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Benchmarking OpenAI datasets

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
| `get-dataset` | Preprocess the AnEM dataset |
| `train` | Train a NER model from the AnEM corpus |
| `evaluate` | Evaluate results for the NER model |
| `openai-preprocess` | Convert from spaCy format into JSONL. |
| `openai-predict` | Fetch zero-shot NER results using Prodigy's GPT-3 integration |
| `openai-evaluate` | Evaluate zero-shot GPT-3 predictions |
| `train-curve` | Train a model at varying portions of the training data |
| `clean-datasets` | Drop the Prodigy dataset that was automatically created during the train-curve command |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `ner` | `get-dataset` &rarr; `train` &rarr; `evaluate` |
| `gpt` | `openai-preprocess` &rarr; `openai-predict` &rarr; `openai-evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/span-labeling-datasets` | Git | The span-labeling-datasets repository that contains loaders for AnEM |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->