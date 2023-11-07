<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Hugging Face Hub integration

With [Hugging Face Hub](https://https://huggingface.co/), you can easily share any trained pipeline with the community. The Hugging Face Hub offers:

- Free model hosting.
- Built-in file versioning, even with very large files, thanks to a git-based approach.
- In-browser widgets to play with the uploaded models.

This uses [`spacy-huggingface-hub`](https://github.com/explosion/spacy-huggingface-hub) to push a packaged pipeline to the Hugging Face Hub, including the `whl` file. This enables using `pip install`ing a pipeline directly from the Hugging Face Hub.


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
| `login` | Log in to Hugging Face and download a model |
| `preprocess` | Convert the data to spaCy's binary format |
| `train` | Train a named entity recognition model |
| `evaluate` | Evaluate the model and export metrics |
| `package` | Package the trained model so it can be installed |
| `push_to_hub` | Push the model to the Hub |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` &rarr; `package` &rarr; `push_to_hub` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/fashion_brands_training.jsonl`](assets/fashion_brands_training.jsonl) | Local | JSONL-formatted training data exported from Prodigy, annotated with `FASHION_BRAND` entities (1235 examples) |
| [`assets/fashion_brands_eval.jsonl`](assets/fashion_brands_eval.jsonl) | Local | JSONL-formatted development data exported from Prodigy, annotated with `FASHION_BRAND` entities (500 examples) |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->