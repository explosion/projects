# Text classification: Predicting whether a GitHub issue is about docs

This directory contains the datasets and scripts for an example project using [Prodigy](https://prodi.gy) to train a **binary text classifier with exclusive classes** to predict whether a GitHub issue title is about documentation.

We've limited our experiments to spaCy, but you can use the annotations in any other text classification system instead. **If you run the experiments, please let us know!** Feel free to submit a pull request with your scripts.

## 🧮 Results

| Model                                                                                                   |  F-Score | # Examples |
| ------------------------------------------------------------------------------------------------------- | -------: | ---------: |
| **[spaCy](https://spacy.io)**<br />blank                                                                |     88.8 |        661 |
| **[spaCy](https://spacy.io)**<br /> [`en_vectors_web_lg`](https://spacy.io/models/en#en_vectors_web_lg) | **91.9** |        661 |

## 📚 Data

Labelling the data with [Prodigy](https://prodi.gy) took about two hours and was done manually using the binary classification interface. The raw text was sourced from the from the [GitHub API](https://developer.github.com/v3/) using the search queries `"docs"`, `"documentation"`, `"readme"` and `"instructions"`.

| File                                                       | Count | Description                                           |
| ---------------------------------------------------------- | ----: | ----------------------------------------------------- |
| [`docs_issues_training.jsonl`](docs_issues_training.jsonl) |   661 | Training data annotated with `DOCUMENTATION` label.   |
| [`docs_issues_eval.jsonl`](docs_issues_eval.jsonl)         |   500 | Evaluation data annotated with `DOCUMENTATION` label. |

### Training and evaluation data format

The training and evaluation datasets are distributed in Prodigy's simple JSONL (newline-delimited JSON) format. Each entry contains a `"text"`, the `"label"` and an `"answer"` (`"accept"` if the label applies, `"reject"` if it doesn't apply). Here are two simplified example entries:

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

We also trained [a model](https://autocat.apps.allenai.org/?uid=d9cd6f8c-8f1d-4367-b1ae-b6264bfe2cda) using Allen AI's [Autocat](https://autocat.apps.allenai.org) app (a web-based tool for training, visualizing and showcasing spaCy text classification models). You can try out the classifier in real-time and see the updated predictions as you type. You can also evaluate it on your own data, download the model Python package or just `pip install` it with one command to try it locally. [**View model here.**](https://autocat.apps.allenai.org/?uid=d9cd6f8c-8f1d-4367-b1ae-b6264bfe2cda)

To use the JSONL data in Autocat, we added `"labels": ["DOCUMENTATION"]` to all examples with `"answer": "accept"` and `"labels": ["N/A"]` to all examples with `"answer": "reject"`.

## 🎛 Scripts

The [`scripts_spacy.py`](scripts_spacy.py) file includes command line scripts for training and evaluating spaCy models using the data in Prodigy's format. This should let you reproduce our results. We tried to keep the scripts as straightforward as possible. To see the available arguments, you can run `python scripts_spacy.py [command] --help`.

| Command    | Description                                                                          |
| ---------- | ------------------------------------------------------------------------------------ |
| `train`    | Train a model from Prodigy annotations. Will optionally save the best model to disk. |
| `evaluate` | Evaluate a trained model on Prodigy annotations and print the accuracy.              |
