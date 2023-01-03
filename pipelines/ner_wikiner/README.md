<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Named Entity Recognition (WikiNER)

Simple example of downloading and converting source data and training a named entity recognition model. The example uses the WikiNER corpus, which was constructed semi-automatically. The main advantage of this corpus is that it's freely available, so the data can be downloaded as a project asset. The WikiNER corpus is distributed in IOB format, a fairly common text encoding for sequence data. The `corpus` subcommand splits the corpus into training, development and testing partitions, and uses `spacy convert` to convert them into spaCy's binary format. You can then edit the config to try out different settings, and trigger training with the `train` subcommand.

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
| `train` | Train the full pipeline |
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

## 🚀 Accelerate
If you are interested in accelerating this pipeline, give a look at [ner_wikiner_accelerated_with_speedster](https://github.com/explosion/projects/tree/v3/experimental/ner_wikiner_accelerated_with_speedster) pipeline.
