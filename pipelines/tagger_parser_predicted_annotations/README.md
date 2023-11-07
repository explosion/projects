<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Using Predicted Annotations in Subsequent Components

This project shows how to use the predictions from one pipeline component as features for a subsequent pipeline component in **spaCy v3.1+**. In this demo, which trains a parser and a tagger on [`UD_English-EWT`](https://github.com/UniversalDependencies/UD_English-EWT), the `token.dep` attribute from the parser is used as a feature by the tagger. To make the predicted `DEP` available to the tagger during training, `DEP` is added to `[components.tagger.model.tok2vec.embed.attrs]` and `parser` is added to `[training.annotating_components]` in the config. This particular example does not lead to a large difference in performance, but the tagger accuracy improves from to 92.67% to 92.97% with the addition of `DEP`.

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
| `convert` | Convert the data to spaCy's format |
| `train` | Train UD_English-EWT |
| `evaluate` | Evaluate on the test data and save the metrics |
| `package` | Package the trained model so it can be installed |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `convert` &rarr; `train` &rarr; `evaluate` &rarr; `package` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/UD_English-EWT` | Git |  |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
