<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Named Entity Recognition (WikiNER)

Simple example of downloading and converting source data and training a named entity recognition model. The example uses the WikiNER corpus, which was constructed semi-automatically. The main advantage of this corpus is that it's freely available, so the data can be downloaded as a project asset. The WikiNER corpus is distributed in IOB format, a fairly common text encoding for sequence data. The `corpus` subcommand splits the corpus into training, development and testing partitions, and uses `spacy convert` to convert them into spaCy's binary format. You can then edit the config to try out different settings, and trigger training with the `train` subcommand.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[Weasel documentation](https://github.com/explosion/weasel).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `corpus` | Convert the data to spaCy's format |
| `train` | Train the full pipeline |
| `evaluate` | Evaluate on the test data and save the metrics |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `corpus` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/aij-wikiner-en-wp2.bz2` | URL |  |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->

## 🚀 Accelerate
If you are interested in accelerating this pipeline, have a look at [ner_wikiner_speedster](https://github.com/explosion/projects/tree/v3/experimental/ner_wikiner_speedster) pipeline.
