<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Demo floret vectors

Train floret vectors and load them into a spaCy vectors model.

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
| `tokenize-oscar` | Download, tokenize, and sentencize data |
| `train-floret` | Train floret vectors |
| `init-floret-vectors` | Create a floret vectors model |
| `floret-nn` | Demo nearest neighbors for intentional OOV misspelling 'outdooor' |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `tokenize-oscar` &rarr; `train-floret` &rarr; `init-floret-vectors` &rarr; `floret-nn` |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
