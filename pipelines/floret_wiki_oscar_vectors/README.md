<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Train floret vectors from Wikipedia and OSCAR

This project downloads, extracts and preprocesses texts from Wikipedia and
OSCAR and trains vectors with [floret](https://github.com/explosion/floret).

By default, the project trains floret vectors for Macedonian.

Prerequisites:
- a large amount of hard drive space
- a workstation with a good CPU, or a lot of patience

For Macedonian, you'll need ~5GB in `/scratch` and ~1GB in `vectors/`.

Adjust the variables `n_process` and `vector_thread` for your CPU.

## Text Sources

- Wikipedia: https://dumps.wikimedia.org
- OSCAR 2019: https://oscar-corpus.com/post/oscar-2019/

By default the full OSCAR 2019 dataset is loaded in streaming mode. Adjust
`oscar_max_texts` to use a subset of the full dataset, especially for large
languages like English, Spanish, Chinese, Russian, etc. The text lengths are
not consistent, but 1M texts may be ~3-5GB.

## wikiextractor

In order to fix a few bugs and support multiprocessing with spawn, this
project installs a fork of [`wikiextractor`
v3.0.6](https://github.com/attardi/wikiextractor) as wikiextractor v3.0.7a0.
The modifications to wikiextractor v3.0.6 are described in [this
commit](https://github.com/adrianeboyd/wikiextractor/commit/f8b539d46cd67205884d701c1d5fd18eda84825f).

## wikiextractor

In order to fix a few bugs and support multiprocessing with spawn, this
project installs a fork of [`wikiextractor`
v3.0.6](https://github.com/attardi/wikiextractor) as wikiextractor v3.0.7a0.
The modifications to wikiextractor v3.0.6 are described in [this
commit](https://github.com/adrianeboyd/wikiextractor/commit/f8b539d46cd67205884d701c1d5fd18eda84825f).

## floret Parameters

[floret](https://github.com/explosion/floret) has a large number of
parameters and it's difficult to give advice for all configurations, but the
parameters described here are the ones that it makes sense to customize for
any new language and to experiment with initially.

Be aware that if you're using more than one thread, the results of each run
with fastText or floret will be slightly different.

### `vector_minn` / `vector_maxn`

The minimum and maximum character n-gram lengths should be adapted for the
language and writing system. The n-grams should capture common grammatical
affixes like English `-ing`, without making the number of n-grams per word
too large. Very short n-grams aren't meaningful and very long n-grams will be
too sparse and won't be useful for cases with misspellings and noise.

A good rule of thumb is that `maxn` should correspond to the length of the
longest common affix + `1`, so for many languages with alphabets, `minn
5`/`maxn 5` can be a good starting point, similar to the defaults in the
[original fastText vectors](https://fasttext.cc/docs/en/crawl-vectors.html).

For writing systems where one character corresponds to a syllable, shorter
n-grams are typically more suitable. For Korean, where each (normalized)
character is a syllable and most grammatical affixes are 1-2 characters,
`minn 2`/`maxn 3` seems to perform well.

### `vector_bucket`

The bucket size is the number of rows in the floret vector table. For
tagging and parsing, a bucket size of 50k performs well, but larger sizes may
still lead to small improvements. For NER, the performance continues to
improve for bucket sizes up to at least 200k.

In a spaCy pipeline package, 50k 300-dim vectors are ~60MB and 200k 300-dim
vectors are ~230MB.

### `vector_hash_count`

The recommended hash count is `2`, especially for smaller bucket sizes.

Larger hash counts are slower to train with floret and slightly slower in
inference in spaCy, but may lead to slightly improved performance, especially
with larger bucket sizes.

### `vector_epoch`

You may want to reduce the number of epochs for larger training input sizes.

### `vector_min_count`

You may want to increase the minimum word count for larger training input
sizes.

### `vector_lr`

You may need to decrease the learning rate for larger training input sizes to
avoid NaN errors, see:
https://fasttext.cc/docs/en/faqs.html#im-encountering-a-nan-why-could-this-be

### `vector_thread`

Adjust the number of threads for your CPU. With a larger number of threads,
you may need more epochs to reach the same performance.

## Notes

The project does not currently clean up any intermediate files so that it's
possible to resume from any point in the workflow. The overall disk space
could be reduced by cleaning up files after each step, keeping only the final
floret input text file. floret does require the input file to be on disk
during training.

floret always writes the full `.bin` and `.vec` files after training. These
may be 5GB+ each even though the final `.floret` table is much smaller.

Import the floret vectors into a spaCy vectors model with:

```shell
spacy init vectors mk vectors/mk.floret /path/to/mk_vectors_model --mode floret
```


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
| `extract-wikipedia` | Convert Wikipedia XML to JSONL with wikiextractor |
| `tokenize-wikipedia` | Tokenize and sentencize Wikipedia |
| `tokenize-oscar` | Tokenize and sentencize OSCAR dataset |
| `create-input` | Concatenate tokenized input texts |
| `train-floret-vectors` | Train floret vectors |
| `train-fasttext-vectors` | Train fastText vectors |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `extract-wikipedia` &rarr; `tokenize-wikipedia` &rarr; `tokenize-oscar` &rarr; `create-input` &rarr; `train-floret-vectors` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `/scratch/vectors/downloaded/wikipedia/mkwiki-latest-pages-articles.xml.bz2` | URL |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
