<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Using Prodigy's OpenAI recipes for a bio NER task

This project showcases Prodigy's OpenAI recipe for named-entity recognition
using the [Anatomical Entity Mention (AnEM)
dataset](https://aclanthology.org/W12-4304/).  The dataset contains 11
anatomical entities (e.g., *organ*, *tissue*, *cellular component*, etc.)
based from the Common Anatomy Reference Ontology. The dataset statistics (and
some examples) are shown below:

<!-- TODO: insert dataset statistics -->

In this project, we trained a transformer-based NER model and compared it with the zero-shot
predictions of GPT-3. We wanted to test how large language models fare in a specific domain and
suggest ways on how we can leverage them to improve our annotations. 

<!-- TODO: insert zero-shot and supervised learning diagrams -->
<!-- TODO: insert results -->

The transformer and zero-shot pipelines are defined by the `ner` and `gpt` workflows respectively.
In order to run the `gpt` workflow, make sure to [install Prodigy](https://prodi.gy/docs/install) as well
as a few additional Python dependencies:

```
python -m pip install prodigy -f https://XXXX-XXXX-XXXX-XXXX@download.prodi.gy
python -m pip install -r requirements.txt
```

With `XXXX-XXXX-XXXX-XXXX` being your personal Prodigy license key.

Then, [create a new API key from
openai.com](https://platform.openai.com/account/api-keys) or fetch an existing
one. Record the secret key as well as the organization key and make sure these
are available as environmental variables. For instance, set them in a `.env`
file in the root directory:

```
PRODIGY_OPENAI_ORG = "org-..."
PRODIGY_OPENAI_KEY = "sk-..."
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
| `get-dataset` | Preprocess the AnEM dataset |
| `train` | Train a NER model from the AnEM corpus |
| `evaluate` | Evaluate results for the NER model |
| `openai-preprocess` | Convert from spaCy format into JSONL. |
| `openai-predict` | Fetch zero-shot NER results using Prodigy's GPT-3 integration |
| `openai-evaluate` | Evaluate zero-shot GPT-3 predictions |
| `train-curve` | Train a model at varying portions of the training data |
| `clean-datasets` | Drop the Prodigy dataset that was automatically created during the train-curve command |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `ner` | `get-dataset` &rarr; `train` &rarr; `evaluate` |
| `gpt` | `openai-preprocess` &rarr; `openai-predict` &rarr; `openai-evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/span-labeling-datasets` | Git | The span-labeling-datasets repository that contains loaders for AnEM |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->