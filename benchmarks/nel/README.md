<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: NEL Benchmark

Pipeline for benchmarking NEL approaches (incl. candidate generation and entity disambiguation).

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
| `download_mewsli9` | Download Mewsli-9 dataset. |
| `preprocess` | Preprocess test datasets. |
| `download_model` | Download a model with pretrained vectors and NER component. |
| `parse_wiki_dumps` | Parse Wikipedia dumps. This can take a long time! |
| `create_kb` | Create the knowledge base and write it to file. |
| `compile_corpora` | Compile corpora, separated in in train/dev/test sets. |
| `train` | Train a new Entity Linking component. Pass --vars.gpu_id GPU_ID to train with GPU. Training with some datasets may take a long time! |
| `evaluate` | Evaluate on the test set. |
| `delete_wiki_db` | Deletes SQLite database generated in step parse_wiki_dumps with data parsed from Wikidata and Wikipedia dump. |
| `clean` | Remove intermediate files (excluding Wiki resources and database) |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `download_mewsli9` &rarr; `preprocess` &rarr; `download_model` &rarr; `parse_wiki_dumps` &rarr; `create_kb` &rarr; `compile_corpora` &rarr; `train` &rarr; `evaluate` |
| `training` | `create_kb` &rarr; `compile_corpora` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/wiki/wikidata_entity_dump.json.bz2` | URL | Wikidata entity dump. Download can take a long time! |
| `assets/wiki/wikipedia_dump.xml.bz2` | URL | Wikipedia dump. Download can take a long time! |
| `assets/wiki/wikidata_entity_dump_filtered.json.bz2` | URL | Filtered Wikidata entity dump for demo purposes. |
| `assets/wiki/wikipedia_dump_filtered.xml.bz2` | URL | Filtered Wikipedia dump for demo purposes. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

Notes: 
- `svn` is required for downloading the Mewsli-9 dataset.
- The project configuration specifies a complete dump of the English Wikidata and Wikipedia as well as filtered versions. 
  By default only the filtered versions - containing only articles and entities mentioning "New York" or "Boston" - are 
  downloaded and processed.
  If you'd like to work with the complete dumps, make sure to...
  - ...fetch assets with `extra` (`spacy project assets --extra`).
  - ...set `vars.use_filtered_dumps: ""` in `project.yml`.