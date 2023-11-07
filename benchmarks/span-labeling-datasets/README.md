<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Span labeling datasets

This project compiles various NER and more general spancat datasets 
and their converters into the [spaCy format](https://spacy.io/api/data-formats). 
You can use this to try out experiment with `ner` and `spancat`
or to potentially pre-train them for your application.


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
| `preprocess-wnut17` | Canonicalize the WNUT2017 data set for conversion to .spacy. |
| `convert-wnut17-ents` | Convert WNUT17 dataset into the spaCy format |
| `convert-wnut17-spans` | Convert WNUT17 dataset into the spaCy format |
| `inspect-wnut17` | Analyze span-characteristics |
| `unpack-conll` | Decompress ConLL 2002, remove temporary files and change encoding. |
| `preprocess-conll` | Canonicalize the Dutch ConLL data set for conversion to .spacy. |
| `convert-conll-spans` | Convert CoNLL dataset (de, en, es, nl) into the spaCy format |
| `convert-conll-ents` | Convert CoNLL dataset (de, en, es, nl) into the spaCy format |
| `inspect-conll` | Analyze span-characteristics |
| `convert-archaeo-spans` | Convert Dutch Archaeology dataset into the spaCy format |
| `convert-archaeo-ents` | Convert Dutch Archaeology dataset into the spaCy format |
| `inspect-archaeo` | Analyze span-characteristics |
| `clean-archaeo` |  |
| `convert-anem-spans` | Convert AnEM dataset into the spaCy format |
| `convert-anem-ents` | Convert AnEM dataset into the spaCy format |
| `inspect-anem` | Analyze span-characteristics |
| `preprocess-restaurant` | Make MIT Restaurant Review data set format comply with convert. |
| `convert-restaurant-ents` | Convert MIT Restaurant Review data to .spacy format |
| `convert-restaurant-spans` | Convert MIT Restaurant Review dataset into the spaCy format |
| `inspect-restaurant` | Analyze span-characteristics |
| `generate-unseen` | Create unseen entities splits for all preprocessed datasets. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `wnut17` | `preprocess-wnut17` &rarr; `convert-wnut17-ents` &rarr; `convert-wnut17-spans` &rarr; `inspect-wnut17` |
| `conll` | `unpack-conll` &rarr; `preprocess-conll` &rarr; `convert-conll-ents` &rarr; `convert-conll-spans` &rarr; `inspect-conll` |
| `archaeo` | `convert-archaeo-ents` &rarr; `convert-archaeo-spans` &rarr; `clean-archaeo` &rarr; `inspect-archaeo` |
| `anem` | `convert-anem-ents` &rarr; `convert-anem-spans` &rarr; `inspect-anem` |
| `restaurant` | `preprocess-restaurant` &rarr; `convert-restaurant-ents` &rarr; `convert-restaurant-spans` &rarr; `inspect-restaurant` |
| `all` | `preprocess-wnut17` &rarr; `convert-wnut17-ents` &rarr; `convert-wnut17-spans` &rarr; `unpack-conll` &rarr; `convert-conll-spans` &rarr; `convert-conll-ents` &rarr; `convert-archaeo-ents` &rarr; `convert-archaeo-spans` &rarr; `convert-anem-ents` &rarr; `convert-anem-spans` &rarr; `preprocess-restaurant` &rarr; `convert-restaurant-ents` &rarr; `convert-restaurant-spans` &rarr; `inspect-restaurant` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/wnut17-train.iob` | URL | WNUT17 training dataset for Emerging and Rare Entities Task from Derczynski et al. (ACL 2017) |
| `assets/wnut17-test.iob` | URL | WNUT17 test dataset for Emerging and Rare Entities Task from Derczynski et al. (ACL 2017) |
| `assets/wnut17-dev.iob` | URL | WNUT17 dev dataset for Emerging and Rare Entities Task from Derczynski et al. (ACL 2017) |
| `assets/conll.tgz` | URL | ConLL 2002 shared task data from Tjong Kim Sang (ACL 2002) |
| `assets/archaeo.bio` | URL | Dutch Archaeological NER dataset by Alex Brandsen (LREC 2020) |
| `assets/anem-train.iob` | URL | Anatomical Entity Mention Detection training dataset from Ohta et al. (ACL 2012) |
| `assets/anem-test.iob` | URL | Anatomical Entity Mention Detection test dataset from Ohta et al. (ACL 2012) |
| `assets/restaurant-train_raw.iob` | URL | Training data from the MIT Restaurants Review dataset |
| `assets/restaurant-test_raw.iob` | URL | Test data from the MIT Restaurants Review dataset |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
