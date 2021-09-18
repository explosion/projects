<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Polar Component

This example project shows how to implement a simple stateful component to
score docs on semantic poles.

The method here is based on SemAxis from [An et al
2018](https://arxiv.org/abs/1806.05521). The basic idea is that given a set
of word vectors and some seed poles, like "bad-good", it's possible to
calculate reference vectors. The distance of document vectors from those
reference vectors is like a sentiment or polar score of the document. While
not as sophisticated as a trained model, it's easy to test with existing data.

If you use enough poles, you can use the scores as semantic vectors that can
make downstream tasks explainable. This is explored in the SemAxis paper as
well as [Mathew et al 2020](https://arxiv.org/abs/2001.09876), "The Polar
Framework". (Incorporating semantic vectors as features in a spaCy model is
left as an exercise for the reader.) 

**Note:** Because the data is hosted on Kaggle, it can't be automatically
downloaded by `spacy project assets`, so you'll have to download it yourself.
See [the assets section of this README](#assets) for the link.


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
| `evaluate` | Check output on sample data |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/IMDB Dataset.csv` | Local | IMDB Review Corpus. Download from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
