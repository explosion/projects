<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Customizing Pipelines

This project includes a script to help customize your pipelines. It allow you to swap a tok2vec for a transformer component, do the opposite, or merge two pipelines.


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
| `merge` | Merge the two pipelines into one pipeline. |
| `use-transformer` | Use a transformer feature source in a pipeline, keeping listeners updated. Output config. |
| `use-tok2vec` | Use a CNN tok2vec feature source in a pipeline, keeping listeners updated. Output config. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
