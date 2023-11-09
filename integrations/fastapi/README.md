<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: FastAPI integration

Use [FastAPI](https://fastapi.tiangolo.com/) to serve your spaCy models and host modern REST APIs. To start the server, you can run `spacy project run start`. To explore the REST API interactively, navigate to `http://127.0.0.1:5000/docs` in your browser. See the examples for how to query the API using Python or JavaScript.

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
| `serve` | Serve the models via a FastAPI REST API using the given host and port |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `start` | `download` &rarr; `serve` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/data.jsonl` | URL | Selected sentences from the CMU Movie Summary Corpus used for testing |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->

## 🚀 Examples

The [`/examples`](examples) directory includes examples of how to send requests
to your API. The following examples are available:

| File                                                                      | Language             |
| ------------------------------------------------------------------------- | -------------------- |
| [`Python_Test-REST-API.ipynb`](examples/Python_Test-REST-API.ipynb)       | Python               |
| [`Javascript_Test-REST-API.html`](examples/Javascript_Test-REST-API.html) | JavaScript (Vanilla) |
|  [`React_Test-REST-API.html`](examples/React_Test-REST-API.html)          | JavaScript (React)   |
