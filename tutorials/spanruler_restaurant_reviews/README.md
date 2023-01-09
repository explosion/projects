<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Using SpanRuler for rule-based Named Entity Recognition

This example project demonstrates how you can use the
[SpanRuler](https://spacy.io/api/spanruler), a component introduced in spaCy
3.3, for rule-based named entity recognition (NER). In spaCy v3 and below,
this functionality can be achieved via the
[EntityRuler](https://spacy.io/api/entityruler). However, we will start
**deprecating** the `entity_ruler` component in favor of `span_ruler` in v4.

Here, we will be using the **MIT Restaurant dataset** (Liu, et al, 2013) to
determine entities such as *Rating*, *Location*, *Restaurant_Name*,
*Price*, *Dish*, *Amenity*,  and *Cuisine* from restaurant reviews.
Below are a few examples from the training data:

![](figures/example_00.png)
![](figures/example_01.png)
![](figures/example_02.png)

First, we will train an NER-only model and treat it as our baseline. Then, we
will attach the `SpanRuler` component **after the `ner` component** of the
existing pipeline. This setup gives us two pipelines we can compare upon. The
rules for each entity type can be found in the `scripts/rules.py` file.

If we look at the results, we see an increase in performance for the majority
of entities with rules:

|          | NER only  | With Spanruler  |
|----------|-----------|-----------------|
| Price    | 83.72     | **84.88**       |
| Rating   | 77.21     | **77.78**       |
| Hours    | 64.78     | **65.31**       |
| Amenity  | 66.67     | **67.61**       |
| Location | 81.17     | **81.99**       |
| Restaurant_Name| 76.28     | **78.44**       |

Overall, we have better performance for the combined `ner` and `span_ruler`
pipeline with our set of rules.

|           | NER only | With Spanruler |
|-----------|----------|----------------|
| Precision | 77.58    | **78.30**      |
| Recall    | 76.23    | **76.96**      |
| F-score   | 76.90    | **77.62**      |


**Reference**

- J. Liu, P. Pasupat, S. Cyphers, and J. Glass. 2013. Asgard: A portable
architecture for multilingual dialogue systems. In *2013 IEEE International
Conference on Acoustics, Speech and Signal Processing*, pages 8386-8390


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
| `download` | Download a spaCy model with pretrained vectors. |
| `preprocess` | Preprocess the raw IOB, convert them into spaCy format, and split them into train, dev, and test partitions. |
| `train` | Train a baseline NER model. |
| `assemble` | Assemble trained NER pipeline with SpanRuler. |
| `evaluate` | Evaluate each model. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `download` &rarr; `preprocess` &rarr; `train` &rarr; `assemble` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/train_raw.iob` | URL | Training data from the MIT Restaurants Review dataset |
| `assets/test_raw.iob` | URL | Test data from the MIT Restaurants Review dataset |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->