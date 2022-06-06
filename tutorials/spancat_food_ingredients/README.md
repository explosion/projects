<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Spancat annotation in Prodigy

This project showcases how to use Prodigy to annotate data for the Spancat component

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
| `manual` | Mark entity spans in a text by highlighting them and selecting the respective labels. |
| `manual_pattern` | Mark entity spans in a text with patterns. |
| `manual_suggester` | Mark entity spans in a text with suggester rules. |
| `train_spancat` | Train a spancat model. |
| `correct` | Correct entity spans predicted by a trained spancat model. |
| `drop` | Drop the prodigy database defined in the project.yml |
| `export` | Export the database defined in the project.yml to .spaCy files |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/recipes.jsonl` | Local | Extract of the [Food.com Recipe & Review](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) dataset with 25.000 entries |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->