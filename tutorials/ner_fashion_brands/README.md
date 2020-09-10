<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Detecting fashion brands in online comments (Named Entity Recognition)

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

This project uses [`sense2vec`](https://github.com/explosion/sense2vec) and [Prodigy](https://prodi.gy) to bootstrap an NER model to detect fashion brands in [Reddit comments](https://files.pushshift.io/reddit/comments/). For more details, see [our blog post](https://explosion.ai/blog/sense2vec-reloaded#annotation).

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://nightly.spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `preprocess` | Convert the data to spaCy's binary format |
| `train` | Train a named entity recognition model |
| `evaluate` | Evaluate the model and export metrics |
| `package` | Package the trained model so it can be installed |
| `visualize-model` | Visualize the model's output interactively using Streamlit |
| `visualize-data` | Explore the annotated data in an interactive Streamlit app |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://nightly.spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/fashion_brands_training.jsonl`](assets/fashion_brands_training.jsonl) | Local | JSONL-formatted training data exported from Prodigy, annotated with `FASHION_BRAND` entities (1235 examples) |
| [`assets/fashion_brands_eval.jsonl`](assets/fashion_brands_eval.jsonl) | Local | JSONL-formatted development data exported from Prodigy, annotated with `FASHION_BRAND` entities (500 examples) |
| [`assets/fashion_brands_patterns.jsonl`](assets/fashion_brands_patterns.jsonl) | Local | Patterns file generated with `sense2vec.teach` and used to pre-highlight during annotation (100 patterns) |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

---

## 📚 Data

[Labelling the data](https://explosion.ai/blog/sense2vec-reloaded#annotation-bootstrap)
took about 2 hours and was done manually using the patterns to pre-highlight
suggestions. The raw text was sourced from the from the
[r/MaleFashionAdvice](https://www.reddit.com/r/malefashionadvice/) and
[r/FemaleFashionAdvice](https://www.reddit.com/r/femalefashionadvice/)
subreddits.

| File                                                                    | Count | Description                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------------------- | ----: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`fashion_brands_patterns.jsonl`](assets/fashion_brands_patterns.jsonl) |   100 | Match patterns created with [`sense2vec.teach`](https://github.com/explosion/sense2vec/tree/master#recipe-sense2vecteach) and [`sense2vec.to-patterns`](https://github.com/explosion/sense2vec/tree/master#recipe-sense2vecto-patterns). Can be used with spaCy's [`EntityRuler`](https://spacy.io/usage/rule-based-matching#entityruler) for a rule-based baseline and faster NER annotation. |
| [`fashion_brands_training.jsonl`](assets/fashion_brands_training.jsonl) |  1235 | Training data annotated with `FASHION_BRAND` entities.                                                                                                                                                                                                                                                                                                                                         |
| [`fashion_brands_eval.jsonl`](assets/fashion_brands_eval.jsonl)         |   500 | Evaluation data annotated with `FASHION_BRAND` entities.                                                                                                                                                                                                                                                                                                                                       |

<img width="250" src="https://user-images.githubusercontent.com/13643239/69343953-d6eccb00-0c6e-11ea-96ed-ea1833eb3902.png" alt="" align="right">

### Visualize the data and model

The [`visualize_data.py`](scripts/visualize_data.py) script lets you visualize
the training and evaluation datasets with
[displaCy](https://spacy.io/usage/visualizers).

```bash
python -m spacy project run visualize-data
```

The [`visualize_model.py`](scripts/visualize_model.py) script is powered by
[`spacy-streamlit`](https://github.com/explosion/spacy-streamlit) and lets you
explore the trained model interactively.

```bash
python -m spacy project run visualize-model
```

### Training and evaluation data format

The training and evaluation datasets are distributed in Prodigy's simple JSONL
(newline-delimited JSON) format. Each entry contains a `"text"` and a list of
`"spans"` with the `"start"` and `"end"` character offsets and the `"label"` of
the annotated entities. The data also includes the tokenization. Here's a
simplified example entry:

```json
{
  "text": "Bonobos has some long sizes.",
  "tokens": [
    { "text": "Bonobos", "start": 0, "end": 7, "id": 0 },
    { "text": "has", "start": 8, "end": 11, "id": 1 },
    { "text": "some", "start": 12, "end": 16, "id": 2 },
    { "text": "long", "start": 17, "end": 21, "id": 3 },
    { "text": "sizes", "start": 22, "end": 27, "id": 4 },
    { "text": ".", "start": 27, "end": 28, "id": 5 }
  ],
  "spans": [
    {
      "start": 0,
      "end": 7,
      "token_start": 0,
      "token_end": 0,
      "label": "FASHION_BRAND"
    }
  ],
  "_input_hash": -874614165,
  "_task_hash": 2136869442,
  "answer": "accept"
}
```
