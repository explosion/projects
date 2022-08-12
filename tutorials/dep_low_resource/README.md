<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Training a POS tagger and dependency parser for a low-resource language

This project trains a part-of-speech tagger and dependency parser for a
low-resource language such as Tagalog. We will be using the
[TRG](https://universaldependencies.org/treebanks/tl_trg/index.html) and
[Ugnayan](https://universaldependencies.org/treebanks/tl_ugnayan/index.html)
treebanks for this task. Since the number of sentences in each corpus is
small, we'll need to evaluate our model using [10-fold cross
validation](https://universaldependencies.org/release_checklist.html#data-split).
How to implement this split will be demonstrated in this project
(`scripts/kfold.py`). The cross validation results can be seen below.

### 10-fold Cross-validation results

|         | TOKEN_ACC | POS_ACC | MORPH_ACC | TAG_ACC | DEP_UAS | DEP_LAS |
|---------|-----------|---------|-----------|---------|---------|---------|
| TRG     | **1.000**     | **0.843**   | 0.749     | **0.833**   | *80.846**   | **0.554**   |
| Ugnayan | 0.998     | 0.819   | **0.995**     | 0.810   | 0.667   | 0.409   |


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
| `preprocess` | Convert the data to spaCy's format |
| `evaluate-kfold` | Evaluate using k-fold cross validation |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `evaluate-kfold` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/tl_trg-ud-test.conllu` | URL | Treebank data for UD_Tagalog-TRG |
| `assets/tl_ugnayan-ud-test.conllu` | URL | Treebank data for UD_Tagalog-Ugnayan |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->