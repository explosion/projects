<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Example project of creating a novel nlp component to do relation extraction from scratch.

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

This example project shows how to define a custom model and wrap it as a spaCy component so it integrates easily in any `nlp` pipeline.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://nightly.spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command     | Description                                                                        |
| ----------- | ---------------------------------------------------------------------------------- |
| `data`      | Parse the gold-standard annotations from the Prodigy annotations.                  |
| `train_cpu` | Train the REL model on the CPU and evaluate on the dev corpus.                     |
| `train_gpu` | Train the REL model with a Transformer on a GPU and evaluate on the dev corpus.    |
| `evaluate`  | Apply the trained model to some sample text.                                       |
| `clean`     | Remove intermediate files to start data preparation & training from a clean slate. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow  | Steps                                       |
| --------- | ------------------------------------------- |
| `all`     | `data` &rarr; `train_cpu` &rarr; `evaluate` |
| `all_gpu` | `data` &rarr; `train_gpu` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://nightly.spacy.io/api/cli#project-assets)
in the project directory.

| File          | Source | Description                                        |
| ------------- | ------ | -------------------------------------------------- |
| `annotations` | Local  | Gold-standard REL annotations created with Prodigy |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
