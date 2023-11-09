<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Span Categorization in Prodigy

This project shows how to use Prodigy to annotate data for the spancat component

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[Weasel documentation](https://github.com/explosion/weasel).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `download` | Download the required spaCy model. |
| `span_manual` | Mark entity spans in a text by highlighting them and selecting the respective labels. |
| `span_manual_pattern` | Mark entity spans in a text with patterns. |
| `train_spancat` | Train a spancat model. |
| `span_correct` | Correct entity spans predicted by the trained spancat model. |
| `db_drop` | Drop the prodigy database defined in the project.yml |
| `db_export` | Export the database defined in the project.yml to `.spacy` files |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/food_recipes.jsonl`](assets/food_recipes.jsonl) | Local | Extract of the [Food.com Recipe & Review](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) dataset with 25.000 entries. |
| [`assets/instructions.html`](assets/instructions.html) | Local | Example .HTML file for annotation instructions. |
| [`assets/patterns.jsonl`](assets/patterns.jsonl) | Local | Example patterns for pre-selecting spans in text. |
| [`prodigy.json`](prodigy.json) | Local | Example prodigy.json file for using instruction files. |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->