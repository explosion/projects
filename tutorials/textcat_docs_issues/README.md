<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Predicting whether a GitHub issue is about documentation (Text Classification)

This project uses [spaCy](https://spacy.io) with annotated data from [Prodigy](https://prodi.gy) to train a **binary text classifier** to predict whether a GitHub issue title is about documentation. The pipeline uses the component `textcat_multilabel` in order to train a binary classifier using only one label, which can be True or False for each document. An equivalent alternative for a binary text classifier would be to use the `textcat` component with two labels, where exactly one of the two labels is True for each document.

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
| `preprocess` | Convert the data to spaCy's binary format |
| `train` | Train a text classification model |
| `evaluate` | Evaluate the model and export metrics |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/docs_issues_training.jsonl`](assets/docs_issues_training.jsonl) | Local | JSONL-formatted training data exported from Prodigy, annotated with `DOCUMENTATION` (661 examples) |
| [`assets/docs_issues_eval.jsonl`](assets/docs_issues_eval.jsonl) | Local | JSONL-formatted development data exported from Prodigy, annotated with `DOCUMENTATION` (500 examples) |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

## 📚 Data

Labelling the data with [Prodigy](https://prodi.gy) took about two hours and was
done manually using the binary classification interface. The raw text was
sourced from the [GitHub API](https://developer.github.com/v3/) using
the search queries `"docs"`, `"documentation"`, `"readme"` and `"instructions"`.

### Training and evaluation data format

The training and evaluation datasets are distributed in Prodigy's simple JSONL
(newline-delimited JSON) format. Each entry contains a `"text"`, the `"label"`
and an `"answer"` (`"accept"` if the label applies, `"reject"` if it doesn't
apply). Here are two simplified example entries:

```json
{
  "text": "Add FAQ's to the documentation",
  "label": "DOCUMENTATION",
  "answer": "accept"
}
```

```json
{
  "text": "Proposal: deprecate SQTagUtil.java",
  "label": "DOCUMENTATION",
  "answer": "reject"
}
```

### Data creation workflow

```bash
prodigy mark docs_issues_data ./raw_text.jsonl --label DOCUMENTATION --view-id classification
```

<img width="250" src="https://user-images.githubusercontent.com/13643239/69798875-7d3a5280-11d2-11ea-94d2-e04f9e18b69e.png" alt="" align="right">

## 🚘🐱 Live demo and model download

We also trained
[a model](https://autocat.apps.allenai.org/?uid=d9cd6f8c-8f1d-4367-b1ae-b6264bfe2cda)
using Allen AI's [Autocat](https://autocat.apps.allenai.org) app (a web-based
tool for training, visualizing and showcasing spaCy text classification models).
You can try out the classifier in real-time and see the updated predictions as
you type. You can also evaluate it on your own data, download the model Python
package or just `pip install` it with one command to try it locally.
[**View model here.**](https://autocat.apps.allenai.org/?uid=d9cd6f8c-8f1d-4367-b1ae-b6264bfe2cda)

To use the JSONL data in Autocat, we added `"labels": ["DOCUMENTATION"]` to all
examples with `"answer": "accept"` and `"labels": ["N/A"]` to all examples with
`"answer": "reject"`.
