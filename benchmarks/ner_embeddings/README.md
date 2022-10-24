<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Comparing embedding layers in spaCy

This project compares `MultiHashEmbed` with its standard embedding counterpart
`MultiEmbed` for Named Entity Recognition. In order to access the datasets,
you first need to clone the [`spancat-datasets`](https://github.com/explosion/spancat-datasets)
repository:

```sh
# Clone the spancat-datasets repo to access the datasets
git clone git@github.com:explosion/spancat-datasets.git
```

and run the necessary conversion scripts. For example, let's perform the
conversion command for the [Anatomical Entity Mention (AnEM)](http://www.nactem.ac.uk/anatomy/)
corpus:

```sh
# While inside the spancat-datasets repository
spacy project run anem
```

This will generate the spaCy files that you can use for training. Once done,
you **should copy these files to this project**. For example, you can perform
a directory copy in Linux via:

```sh
cp -r spancat-datasets/corpus/ner/*. ner_embeddings/corpus/. 
```

You can now supply the local path to the commands and workflows to
perform experiments in **this project:**

```sh
# Perform experiments in the ner_embeddings project
spacy project run train . --vars.dataset anem
```


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
| `download-models` | Download spaCy models for their word-embeddings. |
| `init-fasttext` | Initialize the FastText vectors. |
| `make-tables` | Pre-compute token-to-id tables for MultiEmbed. |
| `train` | Train NER model. |
| `evaluate` | Evaluate NER model. |
| `evaluate-unseen` | Evaluate NER model on unseen entities. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `setup` | `download-models` &rarr; `init-fasttext` &rarr; `make-tables` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/fasttext.en.gz` | URL | English fastText vectors. |
| `assets/fasttext.es.gz` | URL | Spanish fastText vectors. |
| `assets/fasttext.nl.gz` | URL | Dutch fastText vectors. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->