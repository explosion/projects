<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Part-of-speech Tagging & Dependency Parsing (Universal Dependencies)

This project template lets you train a part-of-speech tagger, morphologizer, lemmatizer and dependency parser from a [Universal Dependencies](https://universaldependencies.org/) corpus. It takes care of downloading the treebank, converting it to spaCy's format and training and evaluating the model. The template uses the [`UD_English-EWT`](https://github.com/UniversalDependencies/UD_English-EWT) treebank by default, but you can swap it out for any other available treebank. Just make sure to adjust the `lang` and treebank settings in the variables below. Use `xx` for multi-language if no language-specific tokenizer is available in spaCy. Note that multi-word tokens will be merged together when the corpus is converted since spaCy does not support multi-word token expansion.


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
| `preprocess` | Convert the data to spaCy's format |
| `train` | Train UD_English-EWT |
| `evaluate` | Evaluate on the test data and save the metrics |
| `package` | Package the trained model so it can be installed |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` &rarr; `package` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/UD_English-EWT` | Git |  |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
