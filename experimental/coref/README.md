<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Training a spaCy Coref Model

This project trains a coreference model for spaCy using OntoNotes.

Before using this project:

1. install spaCy with GPU support - see the [install widget](https://spacy.io/usage)
2. run `pip install -r requirements.txt`
3. modify `project.yml` to set your GPU ID and OntoNotes path (see [Data Preparation](#data-preparation) for details)

After that you can just run `spacy project run all`.

Note that during training you will probably see a warning like `Token indices
sequence length is longer than ...`. This is a rare condition that
`spacy-transformers` handles internally, and it's safe to ignore if it
happens occasionally. For more details see [this
thread](https://github.com/explosion/spaCy/discussions/9277#discussioncomment-1374226).

## Data Preparation

To use this project you need a copy of [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19).

If you have OntoNotes and have not worked with the CoNLL 2012 coreference annotations before, set `vars.ontonotes` in `project.yml` to the local path to OntoNotes. The top level directory should contain directories named `arabic`, `chinese`, `english`, and `ontology`. Then run the following command to prepare the coreference data:

```
spacy project run prep-conll-data
```

After that you can execute `spacy project run all`.

If you already have CoNLL 2012 data prepared and concatenated into one file per split, you can specify the paths to the training, dev, and test files directly in `project.yml`, see the `vars` section. After doing so you can run `spacy project run all`.

## Using the Trained Pipeline

After training the pipeline (or downloading a pretrained version), you can load and use it like this:

```
import spacy
nlp = spacy.load("training/coref")

doc = nlp("John Smith called from New York, he says it's raining in the city.")
# check the word clusters
print("=== word clusters ===")
word_clusters = [val for key, val in doc.spans.items() if key.startswith("coref_head")]
for cluster in word_clusters:
    print(cluster)
# check the expanded clusters
print("=== full clusters ===")
full_clusters = [val for key, val in doc.spans.items() if key.startswith("coref_cluster")]
for cluster in full_clusters:
    print(cluster)
```

The prefixes used here are a user setting, so you can customize them for your
own pipelines.


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
| `prep-ontonotes-data` | Rehydrate the data using OntoNotes |
| `prep-test-data` | Prepare minimal dataset for CI testing. Note this will overwrite train/dev/test data! |
| `preprocess` | Convert the data to spaCy's format |
| `train-cluster` | Train the clustering component |
| `prep-span-data` | Prepare data for the span resolver component. |
| `train-span-resolver` | Train the span resolver component. |
| `assemble` | Assemble all parts into a complete coref pipeline. |
| `eval` | Evaluate model on the test set. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `prep` | `preprocess` |
| `train` | `train-cluster` &rarr; `prep-span-data` &rarr; `train-span-resolver` &rarr; `assemble` |
| `ci-test` | `prep-test-data` &rarr; `train-cluster` &rarr; `prep-span-data` &rarr; `train-span-resolver` &rarr; `assemble` &rarr; `eval` |
| `all` | `preprocess` &rarr; `train-cluster` &rarr; `prep-span-data` &rarr; `train-span-resolver` &rarr; `assemble` &rarr; `eval` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/` | Git | CoNLL-2012 scripts and dehydrated data, used for preprocessing OntoNotes. |
| `assets/litbank` | Git | LitBank dataset. Only used for building data for tests. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
