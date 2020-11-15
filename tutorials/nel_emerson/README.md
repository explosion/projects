<a href="https://www.youtube.com/watch?v=8u57WSXVpmw" target="_blank"><img src="https://user-images.githubusercontent.com/13643239/81293769-216fd180-906e-11ea-9f9c-d9dec9163dcc.png" width="300" height="auto" align="right" /></a>

<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Disambiguation of "Emerson" mentions in sentences (Entity Linking)

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

**This project was created as part of a [step-by-step video tutorial](https://www.youtube.com/watch?v=8u57WSXVpmw).** It uses [spaCy](https://spacy.io)'s entity linking functionality and [Prodigy](https://prodi.gy) to disambiguate "Emerson" mentions in text to unique identifiers from Wikidata. As an example use-case, we consider three different people called Emerson: [an Australian tennis player](https://www.wikidata.org/wiki/Q312545), [an American writer](https://www.wikidata.org/wiki/Q48226), and a [Brazilian footballer](https://www.wikidata.org/wiki/Q215952). [See here](https://github.com/explosion/projects/tree/master/nel-emerson) for the previous scripts for spaCy v2.x.

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
| `download` | Download a spaCy model with pretrained vectors |
| `kb` | Create the Knowledge Base in spaCy and write it to file |
| `corpus` | Create a training and dev set from the manually annotated data |
| `train` | Train a new Entity Linking component |
| `evaluate` | Final evaluation on the dev data and printing the results |
| `setup` | Install dependencies |
| `clean` | Remove intermediate files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `setup` &rarr; `download` &rarr; `kb` &rarr; `corpus` &rarr; `train` &rarr; `evaluate` |
| `training` | `kb` &rarr; `corpus` &rarr; `train` &rarr; `evaluate` |


### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://nightly.spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/emerson_annotated_text.jsonl`](assets/emerson_annotated_text.jsonl) | Local | The annotated data |
| [`assets/entities.csv`](assets/entities.csv) | Local | The entities in the knowledge base |
| [`assets/emerson_input_text.txt`](assets/emerson_input_text.txt) | Local | The original input text |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

## Prodigy annotation

To perform the manual annotation in Prodigy, we have written a custom recipe
[`el_recipe.py`](scripts/el_recipe.py).

As input, we need to provide the Knowledge base `my_kb` and NER pipeline
`my_nlp` that are created with the scripts described in the previous section.
Further, the file [`emerson_input_text.txt`](prodigy/emerson_input_text) lists
30 sentences from Wikipedia containing just the mention "Emerson" and not the
full name. These sentences are then annotated with Prodigy by executing the
command

```bash
prodigy entity_linker.manual emersons_annotated emerson_input_text.txt my_nlp/ my_kb entitites.csv -F el_recipe.py
```

The final results are stored to file with

```bash
prodigy db-out emersons_annotated >> emerson_annotated_text.jsonl
```

This JSONL file is included here as well in the [`assets`](assets) subdirectory
so the scripts can be run without having to (re)do this manual annotation.
