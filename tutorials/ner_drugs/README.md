<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Detecting drug names in online comments (Named Entity Recognition)

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

This project uses [Prodigy](https://prodi.gy) to bootstrap an NER model to detect drug names in [Reddit comments](https://files.pushshift.io/reddit/comments/).

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
| [`assets/drugs_training.jsonl`](assets/drugs_training.jsonl) | Local | JSONL-formatted training data exported from Prodigy, annotated with `DRUG` entities (1477 examples) |
| [`assets/drugs_eval.jsonl`](assets/drugs_eval.jsonl) | Local | JSONL-formatted development data exported from Prodigy, annotated with `DRUG` entities (500 examples) |
| [`assets/drugs_patterns.jsonl`](assets/drugs_patterns.jsonl) | Local | Patterns file generated with `terms.teach` and used to pre-highlight during annotation (118 patterns) |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

## 📚 Data

Labelling the data with [Prodigy](https://prodi.gy) took a few hours and was
done manually using the patterns to pre-highlight suggestions. The raw text was
sourced from the from the [r/opiates](https://www.reddit.com/r/opiates/)
subreddit.

| File                                                  | Count | Description                                                                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------- | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`drugs_patterns.jsonl`](assets/drugs_patterns.jsonl) |   118 | Single-token patterns created with [`terms.teach`](https://prodi.gy/docs/recipes#terms-teach) and [`terms.to-patterns`](https://prodi.gy/docs/recipes#terms-to-patterns). Can be used with spaCy's [`EntityRuler`](https://spacy.io/usage/rule-based-matching#entityruler) for a rule-based baseline and faster NER annotation. |
| [`drugs_training.jsonl`](assets/drugs_eval.jsonl)     |  1477 | Training data annotated with `DRUG` entities.                                                                                                                                                                                                                                                                                   |
| [`drugs_eval.jsonl`](assets/drugs_eval.jsonl)         |   500 | Evaluation data annotated with `DRUG` entities.                                                                                                                                                                                                                                                                                 |

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
  "text": "Idk if that Xanax or ur just an ass hole",
  "tokens": [
    { "text": "Idk", "start": 0, "end": 3, "id": 0 },
    { "text": "if", "start": 4, "end": 6, "id": 1 },
    { "text": "that", "start": 7, "end": 11, "id": 2 },
    { "text": "Xanax", "start": 12, "end": 17, "id": 3 },
    { "text": "or", "start": 18, "end": 20, "id": 4 },
    { "text": "ur", "start": 21, "end": 23, "id": 5 },
    { "text": "just", "start": 24, "end": 28, "id": 6 },
    { "text": "an", "start": 29, "end": 31, "id": 7 },
    { "text": "ass", "start": 32, "end": 35, "id": 8 },
    { "text": "hole", "start": 36, "end": 40, "id": 9 }
  ],
  "spans": [
    {
      "start": 12,
      "end": 17,
      "token_start": 3,
      "token_end": 3,
      "label": "DRUG"
    }
  ],
  "_input_hash": -2128862848,
  "_task_hash": -334208479,
  "answer": "accept"
}
```

### Data creation workflow

1. Create a terminology list using 3 seed terms.
   ```bash
   prodigy terms.teach drugs_terms en_core_web_lg --seeds "heroin, benzos, weed"
   ```
2. Convert the termonology list to patterns.
   ```bash
   prodigy terms.to-patterns drugs_terms > drugs_patterns.jsonl
   ```
3. Manually create the training and evaluation data or use an
   [entity ruler](https://spacy.io/usage/rule-based-matching#entityruler) with
   the patterns to pre-highlight suggestions.
   ```bash
   prodigy ner.manual drugs_data en_core_web_sm ./raw_text.jsonl --label DRUG
   ```
   ```bash
   prodigy ner.make-gold drugs_data ./rule-based-model ./raw_text.jsonl --label DRUG --unsegmented
   ```
