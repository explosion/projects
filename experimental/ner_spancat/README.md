<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Example SpanCategorizer project using Indonesian NER

The SpanCategorizer is a component in **spaCy v3.1+** for assigning labels to contiguous spans of text proposed by a customizable suggester function. Unlike spaCy's EntityRecognizer component, the SpanCategorizer can recognize nested or overlapping spans. It also doesn't rely as heavily on consistent starting and ending words, so it may be a better fit for non-NER span labelling tasks. You do have to write a function that proposes your candidate spans, however. If your spans are often short, you could propose all spans under a certain size. You could also use syntactic constituents such as noun phrases or noun chunks, or matcher rules.

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
| `train` | Train the pipeline |
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
| `assets/nergrit_ner-grit` | Git |  |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
