<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: FastAPI integration

> ⚠️ This project template uses the new [**spaCy v3.0**](https://nightly.spacy.io), which
> is currently available as a nightly pre-release. You can install it from pip as `spacy-nightly`:
> `pip install spacy-nightly`. Make sure to use a fresh virtual environment.

Use [FastAPI](https://fastapi.tiangolo.com/) to serve your spaCy models and host modern REST APIs. To install the dependencies and start the server, you can run `spacy project run start`. To explore the REST API interactively, navigate to `http://127.0.0.1:5000/docs` in your browser. See the examples for how to query the API using Python or JavaScript.

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
| `install` | Install dependencies and download models |
| `serve` | Serve the models via a FastAPI REST API using the given host and port |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://nightly.spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `start` | `install` &rarr; `serve` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://nightly.spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/data.jsonl` | URL | Selected sentences from the CMU Movie Summary Corpus used for testing |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

## 🚀 Examples

The [`/examples`](examples) directory includes examples of how to send requests
to your API. The following examples are available:

| File                                                                      | Language             |
| ------------------------------------------------------------------------- | -------------------- |
| [`Python_Test-REST-API.ipynb`](examples/Python_Test-REST-API.ipynb)       | Python               |
| [`Javascript_Test-REST-API.html`](examples/Javascript_Test-REST-API.html) | JavaScript (Vanilla) |
|  [`React_Test-REST-API.html`](examples/React_Test-REST-API.html)          | JavaScript (React)   |
