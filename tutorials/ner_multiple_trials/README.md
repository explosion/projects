<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Training a named-entity recognition (NER) with multiple trials

This project demonstrates how to train a spaCy pipeline with multiple trials.
It trains a named-entity recognition (NER) model on the WikiNEuRal English
dataset.  Having multiple trials is useful for experiments, especially if we
want to account for variance and *dependency* on a random seed. 

Under the hood, the training script in `scripts/train_with_trials.py`
generates a random seed per trial, and runs the `train` command as usual.  You
can find the trained model per trial in `training/trial_{n}/`.

> **Note**
> Because the WikiNEuRal dataset is large, we're limiting the number of samples in the train
> and dev corpus to 500 for demonstration purposes. You can adjust this by
> overriding `vars.limit_samples`, or setting it to `0` to train on the whole
> training corpus.

At evaluation, you can pass a directory containing all the models for each
trial. This process is demonstrated in `scripts/evaluate_with_trials.py`.
This will then result to multiple `metrics/scores.json` files that you can
summarize.


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
| `preprocess` | Preprocess the WikiNEuRal dataset to remove indices and update delimiters. |
| `convert` | Convert IOB dataset into the spaCy format. |
| `train` | Train a named-entity recognition (NER) model for a multiple number of trials. |
| `evaluate` | Evaluate all models for each trial, then summarize the results. |
| `clean` | Remove cached files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `convert` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/raw-en-wikineural-train.iob` | URL | WikiNEuRal (en) training dataset |
| `assets/raw-en-wikineural-dev.iob` | URL | WikiNEuRal (en) dev dataset |
| `assets/raw-en-wikineural-test.iob` | URL | WikiNEuRal (en) test dataset |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->