<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Named Entity Recognition (WikiNER) accelerated using nebullvm

Modification of the WikiNER pipeline, using a transformer as Tok2Vec component and accelerating it with nebullvm library.

Nebullvm is an open-source project

Further info on the WikiNER pipeline can be found in [this section](https://github.com/explosion/projects/tree/v3/pipelines/ner_wikiner).

## 🚀 install nebullvm

Before running the pipeline it is necessary to install nebullvm. Nebullvm can be easily installed using `pip`:
```bash
pip install nebullvm
```
Extra components needed for inference optimization are installed the first time nebullvm is imported in a new environment. We suggest to directly do it running

```bash
python -c "import nebullvm"
```

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects). Modules needed for 
integrating nebullvm with Thinc and Spacy are defined in `scripts/extra_components.py`. 

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `corpus` | Convert the data to spaCy's format |
| `train` | Train the full pipeline and optimize the transformer model for inference |
| `evaluate` | Evaluate on the test data and save the metrics |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `corpus` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/aij-wikiner-en-wp2.bz2` | URL |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
