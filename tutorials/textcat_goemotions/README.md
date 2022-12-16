<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Categorization of emotions in Reddit posts (Text Classification)

This project uses spaCy to train a text classifier on the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) with options for a pipeline with and without transformer weights. To use the BERT-based config, change the `config` variable in the `project.yml`.

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
| `init-vectors` | Download vectors and convert to model |
| `preprocess` | Convert the corpus to spaCy's format |
| `train` | Train a spaCy pipeline using the specified corpus and config |
| `evaluate` | Evaluate on the test data and save the metrics |
| `package` | Package the trained model so it can be installed |
| `visualize` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` &rarr; `package` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/categories.txt` | URL | The categories to train |
| `assets/train.tsv` | URL | The training data |
| `assets/dev.tsv` | URL | The development data |
| `assets/test.tsv` | URL | The test data |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

## Usage

If you want to use the BERT-based config ([`bert.cfg`](configs/bert.cfg)), make
sure you have `spacy-transformers` installed:

```
pip install spacy-transformers
```

You can choose your GPU by setting the `gpu_id` variable in the
[`project.yml`](project.yml).

### Tuning a hyper-parameter in the config

To change hyperparameters, you can edit the [config](configs) (or create a new
custom config). For instance, you could edit the
`components.textcat.model.tok2vec.encode.width` value, changing it to `32`:

```ini
[components.textcat.model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 32
depth = 4
window_size = 1
maxout_pieces = 3
```

Now you can retrain and reevaluate, and commit the updated config and metrics:

```bash
spacy project run train
spacy project run evaluate
git commit configs/my_new_config.cfg metrics/my_new_config.cfg -m "Scores TODO%"
```

You can also run experiments in a more lightweight way by running `spacy train`
directly and
[overwriting](https://spacy.io/usage/training#config-overrides)
hyperparameters on the command line:

```bash
spacy train \
    configs/my_new_config.cfg \
    --components.textcat.model.tok2vec.encode.width 32
```

### Adding components from another model

Suppose you want to keep all the functionality of the `en_core_web_sm` model
and add the textcat model you just trained. You can do this using without
changing your training config by using [`spacy
assemble`](https://spacy.io/api/cli#assemble) - you'll just need to prepare a
config describing your final pipeline.

A sample config for doing this is included in
`configs/cnn_with_pretrained.cfg`. After training the model in this project,
you can combine it with a pretrained pipeline by running `spacy project run
assemble`, which will save the new pipeline to `cnn_with_pretrained/`. 

To make your own config to combine pipelines, the basic steps are:

1. Include all the components you want in `nlp.pipeline`
2. Add a section for each component, specifying the pipeline to source it from.
3. If you have two components of the same type, specify unqiue component names for each.
4. If necessary, specify `replace_listeners` to bundle a component with its tok2vec.

You can also remove many values related to training - since you aren't running
a training loop with `spacy assemble`, default values are fine.

Let's go over the last two steps in a little more detail.

By default, components have a simple default name in the pipeline, like "ner"
or "textcat". However, if you have two copies of a component, then they need to
have different names. If you need to change the name of a component you can do
that by giving it a different name in `nlp.pipeline`, and specifying the name
in the original pipeline using the `name` value in the section for the
component.

It depends on the pipeline, but often components use a Listener to just look
for a tok2vec (or transformer) to get features from. If the tok2vec in your
final pipeline is from the same pipeline as the component you're adding, then
you don't have to do anything. But if a component has a different tok2vec, you
can bundle a standalone copy of the original tok2vec with the component so that it doesn't use the wrong one.

Here's an example of a component that has a different name than it had
originally, and also uses `replace_listeners`:

```ini
[components.my_ner]
source = "my_pipeline"
# this component was just called "ner" originally
name = "ner"
# and it listened to the "tok2vec" in the original pipeline
replace_listeners = ["model.tok2vec"]
```

In the sample config, since most of our components come from the pretrained
pipeline, we use the tok2vec from that in the pipeline, and replace the
listeners for the textcat component we trained in this project. Exactly what
configuration of tok2vecs and listeners works depends on your pipeline, for
more details see the [docs on shared vs. indepedent embedding layers
](https://spacy.io/usage/embeddings-transformers#embedding-layers).

### Using embeddings from a spaCy package

First, download an existing trained pipeline with word vectors.
The word vectors of this model can then be specified in `paths.vectors`
or `initialize.vectors`.

```bash
spacy download en_core_web_lg
spacy train \
    configs/cnn.cfg \
    --paths.vectors "en_core_web_lg" \
    --components.textcat.model.tok2vec.embed.include_static_vectors true
```

### Making and using new embeddings

Uncomment the asset in your [`project.yml`](project.yml):

```yaml
assets:
  - dest: "assets/vectors.zip"
    url: "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
```

Then download the asset and run the `init-vectors` command:

```bash
spacy project assets
spacy project run init-vectors
```

Use the vectors:

```bash
spacy train \
    configs/cnn.cfg \
    --paths.vectors "assets/en_fasttext_vectors" \
    --components.textcat.model.tok2vec.embed.include_static_vectors true
```
