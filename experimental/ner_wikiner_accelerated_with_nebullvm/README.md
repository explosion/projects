<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Named Entity Recognition (WikiNER) accelerated using nebullvm

This project shows how nebullvm can accelerate spaCy's WikiNER pipeline.

[Nebullvm](https://github.com/nebuly-ai/nebullvm) is an open-source tool designed to accelerate AI inference of deep learning models in a few lines of code. Within the WikiNER pipeline, nebullvm optimizes BERT to achieve the maximum acceleration physically possible on the hardware used.

Further info on the WikiNER pipeline can be found in [this section](https://github.com/explosion/projects/tree/v3/pipelines/ner_wikiner).

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

## 🚀 install nebullvm

Before running the WikiNER pipeline, nebullvm must be installed. Nebullvm can be easily installed using `pip`:
```bash
pip install nebullvm
```
Some of the nebullvm components required for inference optimization are installed when nebullvm is imported into a new environment during the first run of the WikiNER pipeline.

Alternatively, these components can be installed beforehand by running
```bash
python -c "import nebullvm"
```

## ⚡ Acceleration 

When tested, [nebullvm](https://github.com/nebuly-ai/nebullvm) accelerated the WikiNER pipeline between **20%** and **80%** with **no impact on model performance**. The library could further accelerate deep learning model inference by applying more aggressive optimization techniques, which may result in a slight change in model performance. For more information, refer to the [nebullvm documentation](https://github.com/nebuly-ai/nebullvm).

Below are the response times of the WikiNER pipeline in milliseconds (ms).

| Hardware | Original latency [ms] | Nebullvm optimized latency [ms] | Nebullvm speed-up |
| --- | --- | --- | --- |
| **Intel** | 139 | 114 | 1.2x |
| **AMD** | 293 | 162 | 1.8x |
| **Nvidia RTX 3090Ti** | 24.1 | 14.1 | 1.7x |
| **M1 Pro** | 143 | 121 | 1.2x |
