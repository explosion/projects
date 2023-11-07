<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Prodigy annotation tool integration

This project shows how to integrate the [Prodigy](https://prodi.gy) annotation tool (requires **v1.11+**) into your spaCy project template to automatically **export annotations** you've created and **train your model** on the collected data. Note that in order to run this template, you'll need to install Prodigy separately into your environment. For details on how the data was created, check out this [project template](https://github.com/explosion/projects/tree/v3/tutorials/ner_fashion_brands) and [blog post](https://explosion.ai/blog/sense2vec-reloaded#annotation).
> ⚠️ **Important note:** The example in this project uses a separate step `db-in` to export the example annotations into your database, so you can easily run it end-to-end. In your own workflows, you can leave this out and access the given dataset you've annotated directly.

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
| `db-in` | Load data into prodigy (only for example purposes) |
| `data-to-spacy` | Merge your annotations and create data in spaCy's binary format |
| `train_spacy` | Train a named entity recognition model with spaCy |
| `train_prodigy` | Train a named entity recognition model with Prodigy |
| `train_curve` | Train the model with Prodigy by using different portions of training examples to evaluate if more annotations can potentially improve the performance |
| `package` | Package the trained model so it can be installed |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `db-in` &rarr; `data-to-spacy` &rarr; `train_spacy` |
| `all_prodigy` | `db-in` &rarr; `train_prodigy` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/fashion_brands_training.jsonl.jsonl` | Local | JSONL-formatted training data exported from Prodigy, annotated with `FASHION_BRAND` entities (1235 examples) |
| `assets/fashion_brands_eval.jsonl.jsonl` | Local | JSONL-formatted development data exported from Prodigy, annotated with `FASHION_BRAND` entities (500 examples) |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
