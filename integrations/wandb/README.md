<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Weights & Biases integration

Use [Weights & Biases](https://www.wandb.com/) for logging of training experiments. This project template uses the IMDB Movie Review Dataset and includes two workflows: `log` for training a simple text classification model and logging the results to Weights & Biases (works out-of-the-box and only requires the `[training.logger]` to be set in the config) and `parameter-search` for running a hyperparameter search using [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps), running the experiments and logging the results.

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
| `login` | Log in to Weights & Biases |
| `data` | Extract the gold-standard annotations |
| `train` | Train a model using the default config |
| `train-search` | Run customized training runs for hyperparameter search using [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps) |
| `clean` | Remove intermediate files. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `log` | `data` &rarr; `train` |
| `parameter-search` | `data` &rarr; `train-search` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/aclImdb_v1.tar.gz` | URL | Movie Review Dataset for sentiment analysis by Maas et al., ACL 2011. |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
