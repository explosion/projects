# NER: detecting fashion brands in online comments

This directory contains the datasets and scripts for an example project using [`sense2vec`](https://github.com/explosion/sense2vec) and [Prodigy](https://prodi.gy) to bootstrap an NER model to detect fashion brands in [Reddit comments](https://files.pushshift.io/reddit/comments/). For more details, see [our blog post](https://explosion.ai/blog/sense2vec-reloaded#annotation).

We've limited our experiments to spaCy, but you can use the annotations in any other NER system instead. You can likely beat spaCy's scores using a system based on a large transformer model. **If you run the experiments, please let us know!** Feel free to submit a pull request with your scripts.

## 🧮 Results

| Model                                                                                                                         |  F-Score | Precision | Recall | wps CPU | wps GPU | # Examples |
| ----------------------------------------------------------------------------------------------------------------------------- | -------: | --------: | -----: | ------: | ------: | ---------: |
| **Rule-based baseline**<br />[`fashion_brands_patterns.jsonl`](fashion_brands_patterns.jsonl)                                 |     48.4 |      96.3 |   32.4 |    130k |    130k |          0 |
| **[spaCy](https://spacy.io)**<br />blank                                                                                      |     65.7 |      77.3 |   57.1 |     13k |     72k |       1235 |
| **[spaCy](https://spacy.io)**<br /> [`en_vectors_web_lg`](https://spacy.io/models/en#en_vectors_web_lg)                       |     73.4 |      81.5 |   66.8 |     13k |     72k |       1235 |
| **[spaCy](https://spacy.io)**<br /> [`en_vectors_web_lg`](https://spacy.io/models/en#en_vectors_web_lg) + tok2vec<sup>1</sup> | **82.1** |      83.5 |   80.7 |      5k |     68k |       1235 |

1. Representations trained on 1 billion words from Reddit comments using [`spacy pretrain`](https://spacy.io/api/cli#pretrain) predicting the `en_vectors_web_lg` vectors (~8 hours on GPU). Download: [`tok2vec_cd8_model289.bin`](https://github.com/explosion/projects/releases/download/tok2vec/tok2vec_cd8_model289.bin)

## 📚 Data

[Labelling the data](https://explosion.ai/blog/sense2vec-reloaded#annotation-bootstrap) took about 2 hours and was done manually using the patterns to pre-highlight suggestions. The raw text was sourced from the from the [r/MaleFashionAdvice](https://www.reddit.com/r/malefashionadvice/) and [r/FemaleFashionAdvice](https://www.reddit.com/r/femalefashionadvice/) subreddits.

| File                                                             | Count | Description                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------- | ----: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`fashion_brands_patterns.jsonl`](fashion_brands_patterns.jsonl) |   100 | Match patterns created with [`sense2vec.teach`](https://github.com/explosion/sense2vec/tree/master#recipe-sense2vecteach) and [`sense2vec.to-patterns`](https://github.com/explosion/sense2vec/tree/master#recipe-sense2vecto-patterns). Can be used with spaCy's [`EntityRuler`](https://spacy.io/usage/rule-based-matching#entityruler) for a rule-based baseline and faster NER annotation. |
| [`fashion_brands_training.jsonl`](fashion_brands_training.jsonl) |  1235 | Training data annotated with `FASHION_BRAND` entities.                                                                                                                                                                                                                                                                                                                                         |
| [`fashion_brands_eval.jsonl`](fashion_brands_eval.jsonl)         |   500 | Evaluation data annotated with `FASHION_BRAND` entities.                                                                                                                                                                                                                                                                                                                                       |

<img width="250" src="https://user-images.githubusercontent.com/13643239/69343953-d6eccb00-0c6e-11ea-96ed-ea1833eb3902.png" alt="" align="right">

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

## 🎛 Scripts

The [`scripts_spacy.py`](scripts_spacy.py) file includes command line scripts for training and evaluating spaCy models using the data in Prodigy's format. This should let you reproduce our results. We tried to keep the scripts as straightforward as possible. To see the available arguments, you can run `python scripts_spacy.py [command] --help`.

| Command    | Description                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `train`    | Train a model from Prodigy annotations. Will optionally save the best model to disk.                                                  |
| `evaluate` | Evaluate a trained model on Prodigy annotations and print the accuracy.                                                               |
| `wps`      | Measure the processing speed in words per second. It's recommended to use a larger corpus of raw text here, e.g. a few million words. |
