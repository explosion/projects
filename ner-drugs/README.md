# NER: detecting drug names in online comments

This directory contains the datasets and scripts for an example project using [Prodigy](https://prodi.gy) to bootstrap an NER model to detect drug names in [Reddit comments](https://files.pushshift.io/reddit/comments/).

We've limited our experiments to spaCy, but you can use the annotations in any other NER system instead. You can likely beat spaCy's scores using a system based on a large transformer model. **If you run the experiments, please let us know!** Feel free to submit a pull request with your scripts.

## 🧮 Results

| Model                                                                                                                         |  F-Score | Precision | Recall | wps CPU | wps GPU | # Examples |
| ----------------------------------------------------------------------------------------------------------------------------- | -------: | --------: | -----: | ------: | ------: | ---------: |
| **Rule-based baseline**<br />[`drugs_patterns.jsonl`](drugs_patterns.jsonl)                                                   |     42.5 |      83.7 |   28.5 |    130k |    130k |          0 |
| **[spaCy](https://spacy.io)**<br />blank                                                                                      |     74.9 |      76.4 |   73.5 |     13k |     72k |       1477 |
| **[spaCy](https://spacy.io)**<br /> [`en_vectors_web_lg`](https://spacy.io/models/en#en_vectors_web_lg)                       |     79.9 |      75.7 |   84.5 |     13k |     72k |       1477 |
| **[spaCy](https://spacy.io)**<br /> [`en_vectors_web_lg`](https://spacy.io/models/en#en_vectors_web_lg) + tok2vec<sup>1</sup> | **80.6** |      76.2 |   85.6 |      5k |     68k |       1477 |

1. Representations trained on 1 billion words from Reddit comments using [`spacy pretrain`](https://spacy.io/api/cli#pretrain) predicting the `en_vectors_web_lg` vectors (~8 hours on GPU). Download: [`tok2vec_cd8_model289.bin`](https://github.com/explosion/projects/releases/download/tok2vec/tok2vec_cd8_model289.bin)

## 📚 Data

Labelling the data with [Prodigy](https://prodi.gy) took a few hours and was done manually using the patterns to pre-highlight suggestions. The raw text was sourced from the from the [r/opiates](https://www.reddit.com/r/opiates/) subreddit.

| File                                           | Count | Description                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------------- | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`drugs_patterns.jsonl`](drugs_patterns.jsonl) |   118 | Single-token patterns created with [`terms.teach`](https://prodi.gy/docs/recipes#terms-teach) and [`terms.to-patterns`](https://prodi.gy/docs/recipes#terms-to-patterns). Can be used with spaCy's [`EntityRuler`](https://spacy.io/usage/rule-based-matching#entityruler) for a rule-based baseline and faster NER annotation. |
| [`drugs_training.jsonl`](drugs_eval.jsonl)     |  1477 | Training data annotated with `DRUG` entities.                                                                                                                                                                                                                                                                                   |
| [`drugs_eval.jsonl`](drugs_eval.jsonl)         |   500 | Evaluation data annotated with `DRUG` entities.                                                                                                                                                                                                                                                                                 |

<img width="250" src="https://user-images.githubusercontent.com/13643239/69343764-7eb5c900-0c6e-11ea-8efd-880048cba6b4.png" alt="" align="right">

### Visualize the data

The [`streamlit_visualizer.py`](streamlit_visualizer.py) script lets you visualize the training and evaluation datasets with [displaCy](https://spacy.io/usage/visualizers).

```bash
pip install streamlit
streamlit run streamlit_visualizer.py
```

### Training and evaluation data format

The training and evaluation datasets are distributed in Prodigy's simple JSONL (newline-delimited JSON) format. Each entry contains a `"text"` and a list of `"spans"` with the `"start"` and `"end"` character offsets and the `"label"` of the annotated entities. The data also includes the tokenization. Here's a simplified example entry:

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
3. Manually create the training and evaluation data or use an [entity ruler](https://spacy.io/usage/rule-based-matching#entityruler) with the patterns to pre-highlight suggestions.
   ```bash
   prodigy ner.manual drugs_data en_core_web_sm ./raw_text.jsonl --label DRUG
   ```
   ```bash
   prodigy ner.make-gold drugs_data ./rule-based-model ./raw_text.jsonl --label DRUG --unsegmented
   ```

## 🎛 Scripts

The [`scripts_spacy.py`](scripts_spacy.py) file includes command line scripts for training and evaluating spaCy models using the data in Prodigy's format. This should let you reproduce our results. We tried to keep the scripts as straightforward as possible. To see the available arguments, you can run `python scripts_spacy.py [command] --help`.

| Command    | Description                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `train`    | Train a model from Prodigy annotations. Will optionally save the best model to disk.                                                  |
| `evaluate` | Evaluate a trained model on Prodigy annotations and print the accuracy.                                                               |
| `wps`      | Measure the processing speed in words per second. It's recommended to use a larger corpus of raw text here, e.g. a few million words. |
