<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Training a named-entity recognition (NER) with multiple trials

This project demonstrates how to train a spaCy pipeline with multiple trials.
Having multiple trials is useful for experiments, especially if we want to
account for variance and *dependency* on a random seed. 

Under the hood, the training script in `scripts/train_with_trials.py`
generates a random seed per trial, and runs the `train` command as usual. 
During evaluation, we pass a directory containing all the models for each
trial and then summarize the results afterwards. You can see the implementation 
in `scripts/evaluate_with_trials.py`.


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
| `preprocess` | Preprocess the ConLL 2003 dataset to remove indices and update delimiters. |
| `convert` | Convert IOB dataset into the spaCy format. |
| `train` | Train a named-entity recognition (NER) model for a multiple number of trials. |
| `evaluate` | Evaluate all models for each trial, then summarize the results. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `convert` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/raw-en-conll-train.iob` | URL | CoNLL 2003 (en) training dataset |
| `assets/raw-en-conll-dev.iob` | URL | CoNLL 2003 (en) dev dataset |
| `assets/raw-en-conll-test.iob` | URL | CoNLL 2003 (en) test dataset |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->