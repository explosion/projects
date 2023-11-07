<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Textcat performance benchmarks

Benchmarking different textcat architectures on different datasets.

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
| `data` | Extract the datasets from their archives. |
| `train` | Run customized training runs: 3 textcat architectures trained on 2 datasets. |
| `summarize` | Summarize the results from the runs and print the best & last scores for each run. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `data` &rarr; `train` &rarr; `summarize` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/aclImdb_v1.tar.gz` | URL | Movie Review Dataset by Maas et al., ACL 2011. |
| `assets/dbpedia_csv.tgz` | URL | DBPedia ontology with 14 nonoverlapping classes by Zhang et al., 2015. |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->