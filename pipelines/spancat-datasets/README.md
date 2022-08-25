<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Spancat datasets

This project compiles various spancat datasets and their converters into the
[spaCy format](https://spacy.io/api/data-formats). You can use this in tandem
with the [`spancat-encoders`](https://github.com/explosion/spancat-encoders)
repository to run various experiments on these datasets.


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
| `convert-wnut17-ents` | Convert WNUT17 dataset into the spaCy format |
| `convert-wnut17-spans` | Convert WNUT17 dataset into the spaCy format |
| `clean-wikineural` | Remove unnecessary indices from wikineural data |
| `convert-wikineural-spans` | Convert WikiNeural dataset (de, en, es, nl) into the spaCy format |
| `convert-wikineural-ents` | Convert WikiNeural dataset (de, en, es, nl) into the spaCy format |
| `clean-conll` | Remove unnecessary indices from ConLL data |
| `convert-conll-spans` | Convert CoNLL dataset (de, en, es, nl) into the spaCy format |
| `convert-conll-ents` | Convert CoNLL dataset (de, en, es, nl) into the spaCy format |
| `convert-archaeo-spans` | Convert Dutch Archaeology dataset into the spaCy format |
| `convert-archaeo-ents` | Convert Dutch Archaeology dataset into the spaCy format |
| `convert-anem-spans` | Convert AnEM dataset into the spaCy format |
| `convert-anem-ents` | Convert AnEM dataset into the spaCy format |
| `clean` | Remove intermediary files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `wnut17` | `convert-wnut17-ents` &rarr; `convert-wnut17-spans` |
| `wikineural` | `clean-wikineural` &rarr; `convert-wikineural-ents` &rarr; `convert-wikineural-spans` |
| `conll` | `clean-conll` &rarr; `convert-conll-spans` &rarr; `convert-conll-ents` |
| `archaeo` | `convert-archaeo-ents` &rarr; `convert-archaeo-spans` |
| `anem` | `convert-anem-ents` &rarr; `convert-anem-spans` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/wnut17-train.iob` | URL | WNUT17 training dataset for Emerging and Rare Entities Task from Derczynski et al., 2017 |
| `assets/wnut17-dev.iob` | URL | WNUT17 dev dataset for Emerging and Rare Entities Task from Derczynski et al., 2017 |
| `assets/wnut17-test.iob` | URL | WNUT17 test dataset for Emerging and Rare Entities Task from Derczynski et al., 2017 |
| `assets/raw-en-wikineural-train.iob` | URL | WikiNeural (en) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-en-wikineural-dev.iob` | URL | WikiNeural (en) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-en-wikineural-test.iob` | URL | WikiNeural (en) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-de-wikineural-train.iob` | URL | WikiNeural (de) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-de-wikineural-dev.iob` | URL | WikiNeural (de) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-de-wikineural-test.iob` | URL | WikiNeural (de) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-es-wikineural-train.iob` | URL | WikiNeural (es) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-es-wikineural-dev.iob` | URL | WikiNeural (es) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-es-wikineural-test.iob` | URL | WikiNeural (es) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-nl-wikineural-train.iob` | URL | WikiNeural (nl) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-nl-wikineural-dev.iob` | URL | WikiNeural (nl) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-nl-wikineural-test.iob` | URL | WikiNeural (nl) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-en-conll-train.iob` | URL | CoNLL 2003 (en) training dataset |
| `assets/raw-en-conll-dev.iob` | URL | CoNLL 2003 (en) dev dataset |
| `assets/raw-en-conll-test.iob` | URL | CoNLL 2003 (en) test dataset |
| `assets/raw-de-conll-train.iob` | URL | CoNLL 2003 (de) training dataset |
| `assets/raw-de-conll-dev.iob` | URL | CoNLL 2003 (de) dev dataset |
| `assets/raw-de-conll-test.iob` | URL | CoNLL 2003 (de) test dataset |
| `assets/raw-es-conll-train.iob` | URL | CoNLL 2002 (es) training dataset |
| `assets/raw-es-conll-dev.iob` | URL | CoNLL 2002 (es) dev dataset |
| `assets/raw-es-conll-test.iob` | URL | CoNLL (es) test dataset |
| `assets/raw-nl-conll-train.iob` | URL | CoNLL 2002 (nl) training dataset |
| `assets/raw-nl-conll-dev.iob` | URL | CoNLL 2002 (nl) dev dataset |
| `assets/raw-nl-conll-test.iob` | URL | CoNLL 202 (nl) test dataset |
| `assets/archaeo.bio` | URL | Dutch Archaeological NER dataset by Alex Brandsen (LREC 2020) |
| `assets/anem-train.iob` | URL | Anatomical Entity Mention (AnEM) training corpus containing abstracts and full-text biomedical papers from Ohta et al. (ACL 2012) |
| `assets/anem-test.iob` | URL | Anatomical Entity Mention (AnEM) test corpus containing abstracts and full-text biomedical papers from Ohta et al. (ACL 2012) |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->