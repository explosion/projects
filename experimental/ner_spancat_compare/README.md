<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Comparing SpanCat and NER using a corpus of biomedical literature (GENIA)

This project demonstrates how spaCy's Span Categorization (SpanCat) and
Named-Entity Recognition (NER) perform on different types of entities. Here, we used
a dataset of biomedical literature containing both overlapping and non-overlapping spans.

### About the dataset

[GENIA](http://www.geniaproject.org/genia-corpus) is a dataset containing
biomedical literature from 1,999 Medline abstracts. It contains a collection
of overlapping and hierarhical spans.  To make parsing easier, we will be
using the [pre-constructed IOB
tags](https://github.com/thecharm/boundary-aware-nested-ner/blob/master/Our_boundary-aware_model/data/genia)
from the [Boundary Aware Nested NER
paper](https://aclanthology.org/D19-1034/)
[repository](https://github.com/thecharm/boundary-aware-nested-ner/). Running `debug data` gives us the
following span characteristics:

| Span Type | Span Length | Span Distinctiveness | Boundary Distinctiveness |
|-----------|-------------|----------------------|--------------------------|
| DNA       | 2.807       | 1.447                | 0.796                    |
| protein   | 2.189       | 1.192                | 0.567                    |
| cell_type | 2.094       | 2.350                | 1.051                    |
| cell_line | 3.286       | 1.910                | 1.044                    |
| RNA       | 2.731       | 2.683                | 1.277                    |


### Experiments

Given what we know from the dataset, we will create the following pipelines:

| Pipeline | Description                                                                                                                             | Workflow Name |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------|---------------|
| SpanCat  | Pure Span Categorization for all types of entities. Serves as illustration to demonstrate suggester functions and as comparison to NER. | `spancat` |
| NER      | Named-Entity Recognition for all types of entities. Serves as illustration to compare with the pure SpanCat implementation       | `ner`         |


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
| `install` | Install dependencies |
| `convert` | Convert IOB file into the spaCy format |
| `create-ner` | Split corpus into separate NER datasets for each GENIA label |
| `train-ner` | Train an NER model for each label |
| `train-spancat` | Train a SpanCat model |
| `evaluate-ner` | Evaluate all NER models |
| `evaluate-spancat` | Evaluate SpanCat model |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `install` &rarr; `convert` &rarr; `create-ner` &rarr; `train-ner` &rarr; `train-spancat` &rarr; `evaluate-ner` &rarr; `evaluate-spancat` |
| `spancat` | `install` &rarr; `convert` &rarr; `train-spancat` &rarr; `evaluate-spancat` |
| `ner` | `install` &rarr; `convert` &rarr; `create-ner` &rarr; `train-ner` &rarr; `evaluate-ner` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/train.iob2` | URL | The training dataset for GENIA in IOB format. |
| `assets/dev.iob2` | URL | The evaluation dataset for GENIA in IOB format. |
| `assets/test.iob2` | URL | The test dataset for GENIA in IOB format. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->