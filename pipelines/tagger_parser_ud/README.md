<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Part-of-speech Tagging & Dependency Parsing (Universal Dependencies)

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

This project template lets you train a part-of-speech tagger and dependency parser from a [Universal Dependencies](https://universaldependencies.org/) corpus. It takes care of downloading the treebank, converting it to spaCy's format and training and evaluating the model. The template uses the [`UD_English-EWT`](https://github.com/UniversalDependencies/UD_English-EWT) treebank by default, but you can swap it our for any other available treebank. Just make sure to adjust the `lang` setting in the config or use `xx` for multi-language if no language-specific tokenizer is available in spaCy.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://nightly.spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `convert` | Convert the data to spaCy's format |
| `train` | Train UD_English-EWT |
| `evaluate` | Evaluate on the test data and save the metrics |
| `package` | Package the trained model so it can be installed |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `convert` &rarr; `train` &rarr; `evaluate` &rarr; `package` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://nightly.spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/UD_English-EWT` | Git |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->