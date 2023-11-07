<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Named Entity Recognition (WikiNER) accelerated using speedster

This project shows how `speedster` can accelerate spaCy's WikiNER pipeline.

[Speedster](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/speedster) is an open-source tool designed to accelerate AI inference of deep learning models in a few lines of code. Within the WikiNER pipeline, `speedster` optimizes BERT to achieve the maximum acceleration physically possible on the hardware used.

`Speedster` is built on top of [Nebullvm](https://github.com/nebuly-ai/nebullvm), an open-source framework for building AI-optimization tools.

Further info on the WikiNER pipeline can be found in [this section](https://github.com/explosion/projects/tree/v3/pipelines/ner_wikiner).

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
| `train` | Train the full pipeline and optimize the transformer model for inference |
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

## 🚀 install speedster

Before running the WikiNER pipeline, speedster must be installed. Speedster can be easily installed using `pip`:
```bash
pip install speedster
```
Some of the speedster components required for inference optimization can be installed using the `nebullvm` auto-installer module.

```bash
python -m nebullvm.installers.auto_installer --frameworks torch onnx huggingface --compilers all
```
If you are interested in installing just a part of the compilers supported by `speedster` you can replace the `all` keyword with the wanted compilers. Further info can be found in the [speedster documentation](https://nebuly.gitbook.io/nebuly/speedster/installation).

## ⚡ Acceleration 

When tested, [speedster](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/speedster) accelerated the WikiNER pipeline between **20%** and **80%** with **no impact on model performance**. The library could further accelerate deep learning model inference by applying more aggressive optimization techniques, which may result in a slight change in model performance. For more information, refer to the [speedster documentation](https://nebuly.gitbook.io/nebuly/speedster/get-started).

Below are the response times of the WikiNER pipeline in milliseconds (ms).

| Hardware | Original latency [ms] | Speedster optimized latency [ms] | Speedster speed-up |
| --- | --- | --- | --- |
| **Intel** | 139 | 114 | 1.2x |
| **AMD** | 293 | 162 | 1.8x |
| **Nvidia RTX 3090Ti** | 24.1 | 14.1 | 1.7x |
| **M1 Pro** | 143 | 121 | 1.2x |
