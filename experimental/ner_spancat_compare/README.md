<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Comparing SpanCat and NER using a corpus of medical abstracts with Patient, Intervention, and Outcomes (PIO) annotations

This project demonstrates how spaCy's Span Categorization (SpanCat) and
Named-Entity Recognition (NER) perform on different types of entities. Here, we used
a dataset of medical abstracts containing both overlapping and non-overlapping spans.

### The Evidence-based Medicine NLP (EBM-NLP) Corpus

The [EBM-NLP (Evidence-based Medicine NLP)
corpus](https://ebm-nlp.herokuapp.com/index) contains 5,000 annotated
abstracts of various medical articles describing clinical trials. It includes
three sets of annotations: 

- Patient (P): the population who received the study.
- Intervention (I): the medication, procedure, and diagnostic test done for the patient.
- Outcome (O): what was accomplished from the study (e.g, accurate diagnosis, relieve symptom, etc.)

Due to the nature of these articles, entities tend to overlap one another. In
the example below, the Intervention, *bestatin*, overlaps with the Patient
annotation. Most abstracts often report in this form.

![](static/sample_00.png)

It is also apparent that some entities can be described by noun phrases
instead of proper nouns. This is especially true for the Patient and Outcome
entities&mdash; describing a population ("acute nonlymphocytic leukemia in
adults") or a study's result ("longer remission", "prolonged survival")
involves a collection of words rather than a single noun.

### Experiments

Given what we know from the dataset, we will create the following pipelines:

| Pipeline | Description                                                                                                                             | Workflow Name |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------|---------------|
| SpanCat  | Pure Span Categorization for all types of entities. Serves as illustration to demonstrate suggester functions and as comparison to NER. | `all-spancat` |
| NER      | Named-Entity Recognition only for the Intervention entity. Serves as illustration to compare with the pure SpanCat implementation       | `ner`         |
| Combined | Combines SpanCat and NER to leverage their strengths. Use SpanCat for Participants and Outcomes, then use NER for Interventions         | `combined`    |


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
| `preprocess` | Convert raw inputs into spaCy's binary format |
| `train-spancat` | Train a SpanCat pipeline |
| `clean` | Remove intermediate files |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/ebm_nlp_1_00.tar.gz` | URL | The full dataset containing text files of medical abstracts and their annotations. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->