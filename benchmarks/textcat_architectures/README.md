<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Textcat performance benchmarks

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

Benchmarking different textcat architectures on different datasets.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://nightly.spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `data` | Extract the datasets from their archives. |
| `train` | Run customized training runs: 3 textcat architectures trained on 2 datasets. |
| `summarize` | Summarize the results from the runs and print the best & last scores for each run. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `data` &rarr; `train` &rarr; `summarize` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://nightly.spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/aclImdb_v1.tar.gz` | URL | Movie Review Dataset by Maas et al., ACL 2011. |
| `assets/dbpedia_csv.tgz` | URL | DBPedia ontology with 14 nonoverlapping classes by Zhang et al., 2015. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->