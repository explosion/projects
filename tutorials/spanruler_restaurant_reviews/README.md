<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Using SpanRuler for rule-based Named Entity Recognition

This example project demonstrates how you can use the
[SpanRuler](https://spacy.io/api/spanruler) component for rule-based named
entity recognition (NER). In spaCy v3 and below, this functionality can be
achieved via the [EntityRuler](https://spacy.io/api/entityruler). However, 
we will start **deprecating** the `entity_ruler` component in favor of
`span_ruler` in v4.

Here, we will be using the **MIT Restaurant dataset** (Liu, et al, 2013) to
determine entities such as *Rating*, *Location*, *Restaurant_Name*,
*Price*, *Dish*, *Amenity*,  and *Cuisine* from restaurant reviews.
Below are a few examples from the training data:

![](figures/example_00.png)
![](figures/example_01.png)
![](figures/example_02.png)

First, we will train an NER-only model and treat it as our baseline. Then, we will
attach the `SpanRuler` component **before the `ner` component** of the existing
pipeline. This setup gives us two pipelines we can compare upon.

We will create rules for `Price`, `Rating`, `Hours`, `Amenity`, and `Location`
because they have discernible patterns we can encode. The same cannot be said
for `Restaurant_Name` and `Dish`, so we'll leave them as they are. Here are
some rules we included in the patterns file (`patterns.jsonl`):

| Label  | Pattern / Examples                                    | Description                                                                 |
|--------|-------------------------------------------------------|-----------------------------------------------------------------------------|
| Price  | `cheap(est)?`, `(in)?expensive` | Reviewers do not often give the exact dollar amount but rather describe it. |
| Rating | `good`, `great`, `fancy`                                      | Reviewers often describe the dish rather than giving an exact rating        |
| Rating | `\d(-\|\s)?star(s)?`                                     | Reviewers can also give star ratings (5-star, 3-stars, 2 star) on a dish.   |
| Rating | `(one\|two\|three\|four\|five)\sstar(s)?`                  | Same as above but using words (four star, five star rating) instead of numbers. |
| Rating | `michelin`, `michelin rated`                  | Reviews also mention if a restaurant has a Michelin star. |
| Amenity | `master card`, `take credit card`                  | Amenities mention different payment options. |
| Amenity | `classy`, `clean`                  | Amenities also include adjectives that describe the restaurant. |
| Location | `in harold square`, `airport`                  | Location also mentions nearby landmarks for a restaurant. |

If we look at the results, we see an increase in performance for the majority
of entities with rules:

|          | NER only  | With Spanruler  |
|----------|-----------|-----------------|
| Price    | **83.72** | 82.90           |
| Rating   | 77.21     | **77.78**       |
| Hours    | 64.78     | **64.78**       |
| Amenity  | 66.67     | **66.98**       |
| Location | 81.17     | **81.20**       |

Overall, we have better performance for the combined `ner` and `span_ruler`
pipeline with just a non-exhaustive set of rules.

|           | NER only | With Spanruler |
|-----------|----------|----------------|
| Precision | 77.58    | **77.63**      |
| Recall    | 76.23    | **76.32**      |
| F-score   | 76.90    | **76.97**      |

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
| `preprocess` | Format and process the raw IOB datasets to make them compatible with spaCy convert. |
| `convert` | Convert the data to spaCy's binary format. |
| `split` | Split the train-dev dataset. |
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
| `all` | `download` &rarr; `preprocess` &rarr; `convert` &rarr; `split` &rarr; `train` &rarr; `assemble` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/train_raw.iob` | URL | Training data from the MIT Restaurants Review dataset |
| `assets/test_raw.iob` | URL | Test data from the MIT Restaurants Review dataset |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->