<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# 🪐 Project Templates

[spaCy projects](https://nightly.spacy.io/usage/projects) let you manage and
share **end-to-end spaCy workflows** for different **use cases and domains**,
and orchestrate training, packaging and serving your custom pipelines. You can
start off by cloning a pre-defined project template, adjust it to fit your
needs, load in your data, train a pipeline, export it as a Python package,
upload your outputs to a remote storage and share your results with your team.

> ⚠️ spaCy project templates require the new
> [**spaCy v3.0**](https://nightly.spacy.io), which is currently available as a
> nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.
>
> See the [`master` branch](https://github.com/explosion/projects/tree/master)
> for the previous version of this repo.

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/20/v3.svg?logo=azure-pipelines&style=flat-square&label=build)](https://dev.azure.com/explosion-ai/public/_build?definitionId=20)
[![spaCy](https://img.shields.io/static/v1?label=made%20with%20%E2%9D%A4%20and&message=spaCy&color=09a3d5&style=flat-square)](https://nightly.spacy.io)

## 🗃 Categories

| Name                           | Description                                                                                                                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`pipelines`](pipelines)       | Templates for training NLP pipelines with different components on different corpora.                                                                                                    |
| [`tutorials`](tutorials)       | Templates that work through a specific NLP use case end-to-end.                                                                                                                         |
| [`integrations`](integrations) | Templates showing integrations with third-party libraries and tools for managing your data and experiments, iterating on demos and prototypes and shipping your models into production. |
| [`benchmarks`](benchmarks)     | Templates to reproduce our benchmarks and produce quantifiable results that are easy to compare against other systems or versions of spaCy.                                             |
| [`experimental`](experimental) | Experimental workflows and other cutting-edge stuff to use at your own risk.                                                                                                            |

## 🚀 Quickstart

Projects can be used via the new
[`spacy project`](https://nightly.spacy.io/api/cli#project) CLI. To find out
more about a command, add `--help`. For detailed instructions, see the
[usage guide](https://nightly.spacy.io/usage/projects).

<!-- TODO: update example -->

1. **Clone** the project template you want to use.
   ```bash
   python -m spacy project clone tutorials/ner_fashion_brands
   ```
2. **Fetch assets** (data, weights) defined in the `project.yml`.
   ```bash
   cd ner_fashion_brands
   python -m spacy project assets
   ```
3. **Run a command** defined in the `project.yml`.
   ```bash
   python -m spacy project run preprocess
   ```
4. **Run a workflow** of multiple steps in order.
   ```bash
   python -m spacy project run all
   ```
5. **Adjust** the template for **your specific use case**, load in your own
   data, adjust the settings and model and share the result with your team.

## 👷‍♀️Repository maintanance

To keep the project templates and their documentation up to date, this repo
contains several scripts:

| Script                                                       | Description                                                                                                                                                                                                                       |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`update_docs.py`](.github/update_docs.py)                   | Update all auto-generated docs in the given root. Calls into [`spacy project document`](https://nightly.spacy.io/api/cli#project-document) and only replaces the auto-generated sections, not any custom content before or after. |
| [`update_category_docs.py`](.github/update_category_docs.py) | Update the auto-generated `README.md` in the category directories listing the available project templates.                                                                                                                        |
| [`update_configs.py`](.github/update_configs.py)             | Update and auto-fill all `config.cfg` files included in the repo, similar to [`spacy init fill-config`](https://nightly.spacy.io/api/cli#init-fill-config). Can be used to keep the configs up to date with changes in spaCy.     |
