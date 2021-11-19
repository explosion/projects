<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Universal Dependencies v2.5 Benchmarks

This project template lets you train a tokenizer, lemmatizer, part-of-speech tagger, a morphologizer and a dependency parser from a [Universal Dependencies](https://universaldependencies.org/) corpus (v2.5). It takes care of downloading the treebank, converting it to spaCy's format and training and evaluating the model. The template uses the [`UD_English-EWT`](https://github.com/UniversalDependencies/UD_English-EWT) treebank by default, but you can swap it out for any other available treebank. Just make sure to adjust the `ud_treebank` and `spacy_lang` settings in the config. Use `xx` (multi-language) for `spacy_lang` if a particular language is not supported by spaCy.

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
| `extract` | Extract the data |
| `convert` | Convert the data to spaCy's format |
| `train-tokenizer` | Train tokenizer |
| `train-lemmatizer` | Train lemmatizer |
| `init-labels-transformer` | Initialize labels for transformer components from gold tokenization |
| `train-transformer` | Train transformer |
| `assemble` | Assemble full pipeline |
| `evaluate` | Evaluate on the test data and save the metrics |
| `package` | Package the trained model so it can be installed |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `extract` &rarr; `convert` &rarr; `train-tokenizer` &rarr; `train-lemmatizer` &rarr; `init-labels-transformer` &rarr; `train-transformer` &rarr; `assemble` &rarr; `evaluate` &rarr; `package` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/ud-treebanks-v2.5.tgz` | URL |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
