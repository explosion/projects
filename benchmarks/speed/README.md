<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Project for speed benchmarking of various pretrained models of different NLP libraries.

This project runs various models on unannotated text, to measure the average speed in words per second (WPS). Note that a fair comparison should also take into account the type of annotations produced by each model, and the accuracy scores of the various pretrained NLP tasks. This example project only addresses the speed issue, but can be extended to perform more detailed comparisons on any data.

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
| `download` | Download models |
| `timing_cpu` | Run all timing benchmarks on CPU and add the numbers to output/results.csv |
| `timing_gpu` | Run all timing benchmarks on GPU and add the numbers to output/results.csv |
| `clean` | Remove output file(s) |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `setup` | `download` |
| `benchmark` | `timing_cpu` &rarr; `timing_gpu` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `texts/reddit-100k.jsonl` | URL | The texts to process |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
