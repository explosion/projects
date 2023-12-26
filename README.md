<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# 🪐 Project Templates

[Weasel](https://github.com/explosion/weasel), previously
[spaCy projects](https://spacy.io/usage/projects), lets you manage and share
**end-to-end workflows** for different **use cases and domains**, and
orchestrate training, packaging and serving your custom pipelines. You can start
off by cloning a pre-defined project template, adjust it to fit your needs, load
in your data, train a pipeline, export it as a Python package, upload your
outputs to a remote storage and share your results with your team.

> ⚠️ Weasel project templates require
> [**Weasel**](https://github.com/explosion/weasel), which is also included by
> default with spaCy v3.7+. You can install it from pip with
> `pip install weasel` or conda with `conda install weasel -c conda-forge`. Make
> sure to use a fresh virtual environment.
>
> See the [`master` branch](https://github.com/explosion/projects/tree/master)
> for the previous version of this repo.

[![tests](https://github.com/explosion/projects/actions/workflows/tests.yml/badge.svg)](https://github.com/explosion/projects/actions/workflows/tests.yml)
[![spaCy](https://img.shields.io/static/v1?label=made%20with%20%E2%9D%A4%20and&message=spaCy&color=09a3d5&style=flat-square)](https://spacy.io)

## 🗃 Categories

| Name                           | Description                                                                                                                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`pipelines`](pipelines)       | Templates for training NLP pipelines with different components on different corpora.                                                                                                    |
| [`tutorials`](tutorials)       | Templates that work through a specific NLP use case end-to-end.                                                                                                                         |
| [`integrations`](integrations) | Templates showing integrations with third-party libraries and tools for managing your data and experiments, iterating on demos and prototypes and shipping your models into production. |
| [`benchmarks`](benchmarks)     | Templates to reproduce our benchmarks and produce quantifiable results that are easy to compare against other systems or versions of spaCy.                                             |
| [`experimental`](experimental) | Experimental workflows and other cutting-edge stuff to use at your own risk.                                                                                                            |

## 🚀 Quickstart

Projects can be used via the
[`weasel`](https://github.com/explosion/weasel/blob/main/docs/cli.md) CLI, or
through the [`spacy project`](https://spacy.io/api/cli#project) alias. To find
out more about a command, add `--help`. For detailed instructions, see the
[Weasel documentation](https://github.com/explosion/weasel/tree/main#-documentation)
or [spaCy projects usage guide](https://spacy.io/usage/projects).

1. **Clone** the project template you want to use.
   ```bash
   python -m weasel clone tutorials/ner_fashion_brands
   ```
2. **Install** any project requirements.
   ```bash
   cd ner_fashion_brands
   python -m pip install -r requirements.txt
   ```
3. **Fetch assets** (data, weights) defined in the `project.yml`.
   ```bash
   python -m weasel assets
   ```
4. **Run a command** defined in the `project.yml`.
   ```bash
   python -m weasel run preprocess
   ```
5. **Run a workflow** of multiple steps in order.
   ```bash
   python -m weasel run all
   ```
6. **Adjust** the template for **your specific use case**, load in your own
   data, adjust the settings and model and share the result with your team.

## 👷‍♀️Repository maintanance

To keep the project templates and their documentation up to date, this repo
contains several scripts:

| Script                                                         | Description                                                                                                                                                                                                               |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`update_docs.py`](.github/update_docs.py)                     | Update all auto-generated docs in the given root. Calls into [`spacy project document`](https://spacy.io/api/cli#project-document) and only replaces the auto-generated sections, not any custom content before or after. |
| [`update_category_docs.py`](.github/update_category_docs.py)   | Update the auto-generated `README.md` in the category directories listing the available project templates.                                                                                                                |
| [`update_configs.py`](.github/update_configs.py)               | Update and auto-fill all `config.cfg` files included in the repo, similar to [`spacy init fill-config`](https://spacy.io/api/cli#init-fill-config). Can be used to keep the configs up to date with changes in spaCy.     |
| [`update_projects_jsonl.py`](.github/update_projects_jsonl.py) | Update `projects.jsonl` file in the given root. Should be used at the root level of the repository                                                                                                                            |
