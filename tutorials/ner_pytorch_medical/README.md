<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Detecting entities in Medical Records

This project uses the [i2b2 (n2c2) 2011 Challenge Dataset](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) to bootstrap an NER model to detect entities in Medical Records. It also demonstrates how to anonymize medical records for annotators in [Prodigy](https://prodi.gy).

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
| `preprocess` | Convert the data to spaCy's binary format |
| `train` | Train a custom PyTorch named entity recognition model |
| `train-trf` | Train a custom PyTorch named entity recognition model with transformer |
| `evaluate` | Evaluate the custom PyTorch model and export metrics |
| `package` | Package the trained model so it can be installed |
| `visualize-model` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

## 📚 Data

The main data source is the i2b2 2011 Co-Reference Challenge Dataset. This dataset has annotations for Named Entity Recognition in Medical Records and co-references between each entity. For the purpose of this tutorial, we focus solely on extracting the labeled entities.

There are no defined assets for this project due to the User Agreement for the Dataset. In order to use this data, you must create an account through the [Harvard DBMI Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).

Once you have an account, navigate to the n2c2 NLP Datasets page. https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

Under the 2011 Coreference Challenge Downloads section,Download the Beth Israel Portion, Partners Portion, and Test Data to the `assets/n2c2_2011` folder. 

![Dataset Downloads](img/n2c2_asset_downloads_screenshot.png)

Your `assets/n2c2_2011` should look like after download:

```
assets
└─── n2c2_2011
        i2b2_Beth_Train_Release.tar.gz
        i2b2_Partners_Train_Release.tar.gz
        Task_1.zip
```

Once you have this data downloaded, the `preprocess` project command will build `*.spacy` dataset files for you.

