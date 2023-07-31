<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Enhancing Morphological Analysis with spaCy Pretraining

This project explores the effectiveness of pretraining techniques on morphological analysis (morphologizer) by conducting experiments on multiple languages. The objective of this project is to demonstrate the benefits of pretraining word vectors using domain-specific data on the performance of the morphological analysis. We leverage the OSCAR dataset to pretrain our vectors for tok2vec and utilize the UD_Treebanks dataset to train a morphologizer component. We evaluate and compare the performance of different pretraining techniques and the performance of models without any pretraining.

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
| `install_requirements` | Download and install all requirements |
| `download_oscar` | Download a subset of the oscar dataset |
| `download_model` | Download the specified spaCy model for vector-objective pretraining |
| `extract_ud` | Extract the ud-treebanks data |
| `convert_ud` | Convert the ud-treebanks data to spaCy's format |
| `train` | Train a morphologizer component without pretrained weights and static vectors |
| `evaluate` | Evaluate the trained morphologizer component without pretrained weights and static vectors |
| `train_static` | Train a morphologizer component with static vectors from a pretrained model |
| `evaluate_static` | Evaluate the trained morphologizer component with static weights |
| `pretrain_char` | Pretrain a tok2vec component with the character objective |
| `train_char` | Train a morphologizer component with pretrained weights (character_objective) |
| `evaluate_char` | Evaluate the trained morphologizer component with pretrained weights (character-objective) |
| `pretrain_vector` | Pretrain a tok2vec component with the vector objective |
| `train_vector` | Train a morphologizer component with pretrained weights (vector_objective) |
| `evaluate_vector` | Evaluate the trained morphologizer component with pretrained weights (vector-objective) |
| `train_trf` | Train a morphologizer component without transformer embeddings |
| `evaluate_trf` | Evaluate the trained morphologizer component with transformer embeddings |
| `evaluate_metrics` | Evaluate all experiments and create a summary json file |
| `reset_project` | Reset the project to its original state and delete all training process |
| `reset_training` | Reset the training progress |
| `reset_metrics` | Delete the metrics folder |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `data` | `download_oscar` &rarr; `download_model` &rarr; `extract_ud` &rarr; `convert_ud` |
| `training` | `train` &rarr; `evaluate` |
| `training_static` | `train_static` &rarr; `evaluate_static` |
| `training_char` | `pretrain_char` &rarr; `train_char` &rarr; `evaluate_char` |
| `training_vector` | `pretrain_vector` &rarr; `train_vector` &rarr; `evaluate_vector` |
| `training_trf` | `train_trf` &rarr; `evaluate_trf` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/ud-treebanks-v2.5.tgz` | URL |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->