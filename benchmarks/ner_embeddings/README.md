<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Comparing embedding layers in spaCy

This project contains the code to reproduce the results of the
[Multi hash embeddings in spaCy](https://arxiv.org/abs/2212.09255) technical report by Explosion.
The `project.yml` provides commands to download and preprocess the data sets as well as to
run the training and evaluation procedures. Different configuration of `vars` correspond
to different experiments in the report.

There are a few scripts included that were used during the technical report writing process
to run experiments in bulk and summarize the results. 
The `scripts/run_experiments.py` runs multiple experiments one after the other
by constructing and running `spacy project run` commands. The module
`scipts/collate_results.py` summarizes the results of the same trials with multiple seeds.
Finally, `scripts/plot_results.py` was used to produce the visualizations in the report.
These are all small command line apps and you can learn more about the usage as usual with the
`--help` flag.

The `rows` argument for the `train-adjusted-rows` command is provided as a list and 
this may lead to errors on Windows machines. Unfortunately, this might lead not being able to
reproduce the `MultiHashEmbed (adjusted)` experiments from the paper on Windows using `run_experiment.py`.
This is due to known issue with handling quotes on Windows and is something we are looking into.
The config files can be edited by manually or in some other way to adjust the number of rows for 
the hash embedding layers. We apologize for the inconvenience.


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
| `prepare-datasets` | Download and preprocess all available data sets using the span-labeling-datasets project. |
| `download-models` | Download spaCy models for their word-embeddings. |
| `init-fasttext` | Initialize the FastText vectors. |
| `make-tables` | Pre-compute token-to-id tables for MultiEmbed. |
| `init-labels` | Initialize labels first before training |
| `train` | Train NER model. |
| `train-adjust-rows` | Train NER model with adjustable number of rows. |
| `train-hash` | Train NER model with different number of hash functions. (only works with the multifewerhashembed.cfg) |
| `evaluate` | Evaluate NER model. |
| `evaluate-seen-only` | Evaluate NER model on the dev and tests sets only considering entities that appear in the training set. |
| `evaluate-unseen-only` | Evaluate NER model on the dev and tests sets only considering entities that did not appear in the training set. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `setup` | `download-models` &rarr; `init-fasttext` &rarr; `prepare-datasets` &rarr; `make-tables` |
| `trial` | `init-labels` &rarr; `train` &rarr; `evaluate` &rarr; `evaluate-seen-only` &rarr; `evaluate-unseen-only` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/fasttext.en.gz` | URL | English fastText vectors. |
| `assets/fasttext.es.gz` | URL | Spanish fastText vectors. |
| `assets/fasttext.nl.gz` | URL | Dutch fastText vectors. |
| `span-labeling-datasets` | Git |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
