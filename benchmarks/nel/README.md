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
| `download_model` | Download a model with pretrained vectors and NER component. |
| `wikid_clone` | Clone `wikid` to prepare Wiki database and `KnowledgeBase`. |
| `preprocess` | Preprocess and clean corpus data. |
| `wikid_download_assets` | Download Wikipedia dumps. This can take a long time if you're not using the filtered dumps! |
| `wikid_parse` | Parse Wikipedia dumps. This can take a long time if you're not using the filtered dumps! |
| `wikid_create_kb` | Create the knowledge base and write it to file. |
| `parse_corpus` | Parse corpus to generate entity and annotation lookups used for corpora compilation. |
| `compile_corpora` | Compile corpora, separated in train/dev/test sets. |
| `train` | Train a new Entity Linking component. Pass --vars.gpu_id GPU_ID to train with GPU. Training with some datasets may take a long time! |
| `evaluate` | Evaluate on the test set. |
| `compare_evaluations` | Compare available set of evaluation runs. |
| `delete_wiki_db` | Deletes SQLite database generated in step wiki_parse with data parsed from Wikidata and Wikipedia dump. |
| `clean` | Remove intermediate files for specified dataset and language (excluding Wiki resources and database). |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `download_mewsli9` &rarr; `download_model` &rarr; `wikid_clone` &rarr; `preprocess` &rarr; `wikid_download_assets` &rarr; `wikid_parse` &rarr; `wikid_create_kb` &rarr; `parse_corpus` &rarr; `compile_corpora` &rarr; `train` &rarr; `evaluate` &rarr; `compare_evaluations` |
| `training` | `train` &rarr; `evaluate` |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

Notes: 
> **Warning**: Parts of this project are currently not platform-agnostic and run only on Linux. Making the entire 
> project work cross-platform is on our todo list. 
- `svn` is required for downloading the Mewsli-9 dataset.
- The project configuration specifies a complete dump of the English Wikidata and Wikipedia as well as filtered versions. 
  By default only the filtered versions - containing only articles and entities mentioning "New York" or "Boston" - are 
  downloaded and processed.
  If you'd like to work with the complete dumps, make sure to...
  - ...fetch assets with `extra` (`spacy project assets --extra`).
  - ...set `vars.use_filtered_dumps: ""` in `project.yml`.