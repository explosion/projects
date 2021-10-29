<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Detecting people entities in tweets (Named Entity Recognition)

This project demonstrates how to improve spaCy's pretrained models by
augmenting the training data and adapting it to a different domain.

**Weak supervision** is the practice of labelling a dataset using imprecise
annotators. Instead of labelling them manually by ourselves (which may take
time and effort), we can just encode business rules or heuristics to do the
labelling for us. 

Of course, these rules don't capture all the nuances a human annotator can.
So weak supervision's idea is to **pool all these naive annotators
together**, and come up with a unified annotator that can hopefully understand
those nuances. The pooling is done via a Hidden Markov Model (HMM). 

Once we have the unified annotator, we re-annotate our entire training dataset,
and use that for our downstream tasks, which in this case, is finetuning spaCy's
`en_core_web_lg` model. 

### Using `skweak` as our weak supervision framework

In this project, we will be using `skweak` as our weak supervision framework.
It contains primitives to define our own labelling functions, and it's
well-integrated to spaCy! These labelling functions can then be defined as
one of the following:

| Annotator Type | What it does                                                                                                                                                                                                                |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model          | makes use of existing models trained from other datasets. Here we can also use spaCy's pretrained models like `en_core_web_lg`                                                                                              |
| Gazetteer      | a list of entities that an annotator can look up from. This will always depend on your use-case.                                                                                                                            |
| Heuristics     | can be used to encode business rules based on one's domain. You can check for tokens that fall under a specific rule (`TokenConstraintAnnotator`), or look for spans of entities in a sentence (`SpanConstraintAnnotator`). |

The unified model is then done using a Hidden Markov Model, in this case, via
the [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/) pacakge. Once we
have our final model, we re-annotate the dataset and resume training via
`spacy train`.

### Weak supervision in practice 

For our dataset, we have a collection of tweets from the [TweetEval
Benchmark](https://github.com/cardiffnlp/tweeteval). Because tweets differ in
structure and form compared to the datasets our original spaCy models were
trained on, it makes sense to adapt the domain. Here are some annotators I
created:

| Annotator                                 | Annotator Type | Heuristic                                                                                                                              |
|-------------------------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Model-based annotator from en_core_web_lg | Model          | We can reuse spaCy models to bootstrap our annotation process. By doing this, we don't have to start entirely from scratch.            |
| Broad Twitter Corpus model                | Model          | Because our dataset is made of tweets, it makes sense to bootstrap from a language model trained on the same domain.                   |
| List of crunchbase personalities          | Gazetteer      | Obtaining a list of tech  personalities is timely, especially due to the context of some tweets.                                       |
| Proper names annotator                    | Heuristics     | A simple heuristic that checks if two proper names are joined by an infix (e.g. Vincent **van** Gogh, Mark **de** Castro, etc.)        |
| Full names annotator                      | Heuristics     | A simple heuristic for finding full names. It checks if a first name exists in a list of names, and is also followed by a proper name. |


There are also some annotators that I've tried, but removed due to its
detrimental effect to our evaluation scores:

| Annotator                            | Annotator Type | Heuristic                                                                                                                                      |
|--------------------------------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| List of personalities from Wikipedia | Gazetteer      | Ideally, we want to include all person entities from a bigger database. However, it detects "you" as a `PERSON`, which affected our Precision. |
| Name suffix annotator                | Heuristic      | I wanted to capture names with "iii", "IV", etc., but it gives lower precision due to it detecting roman numerals that aren't part of names.   |


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
| `decompress` | Decompress relevant assets that will be used latter by our weak supervision model |
| `augment` | Augment an input dataset via weak supervision then split it into training and evaluation datasets |
| `train` | Train a named entity recognition model |
| `train-with-augmenter` | Train a named entity recognition model with a spaCy built-in augmenter |
| `evaluate` | Evaluate various experiments and export the computed metrics |
| `package` | Package the trained model so it can be installed |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `install` &rarr; `preprocess` &rarr; `decompress` &rarr; `augment` &rarr; `train` &rarr; `train-with-augmenter` &rarr; `evaluate` |
| `setup` | `install` &rarr; `preprocess` &rarr; `decompress` |
| `finetune` | `augment` &rarr; `train` &rarr; `train-with-augmenter` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/train_raw.jsonl`](assets/train_raw.jsonl) | Local | The training dataset to augment. Later we'll divide this further to create a dev set |
| [`assets/test_annotated.jsonl`](assets/test_annotated.jsonl) | Local | The held-out test dataset to evaluate our output against. |
| `assets/btc.tar.gz` | URL | A model trained on the Broad Twitter Corpus to help in annotation (63.7 MB) |
| `assets/crunchbase.json.gz` | URL | A list of crunchbase entities to aid the gazetteer annotator (8.56 MB) |
| `assets/wikidata_tokenised.json.gz` | URL | A list of wikipedia entities to aid the gazetteer annotator (21.1 MB) |
| `assets/first_names.json` | URL | A list of first names to help our heuristic annotator |
| `assets/en_orth_variants.json` | URL | Orth variants to use for data augmentation |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
