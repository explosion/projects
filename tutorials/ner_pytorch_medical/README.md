<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Detecting entities in Medical Records with PyTorch

This project uses the [i2b2 (n2c2) 2011 Challenge Dataset](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) to bootstrap a PyTorch NER model to detect entities in Medical Records. It also demonstrates how to anonymize medical records for annotators in [Prodigy](https://prodi.gy).

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

---

## PyTorch + spaCy

There are several spaCy project examples that show the integration of  Transformers in spaCy models. These examples already use PyTorch through the spacy-transformers as to encode spaCy docs as vector representations while using Thinc as task-specific model heads. 

The goal for this tutorial is to show how to use a PyTorch (or other ML framework) end-to-end, from encoding vector representations, to building a task specific model head for Named Entity Recognition.

For more details on using external ML frameworks in spaCy see:
https://spacy.io/usage/layers-architectures#frameworks


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

The data is separated into the following splits:


| Name | Description | N Examples |
|------|-------------|------------|
| train.spacy | 80% of official train data (combining Beth and Partners training data splits) | 200 |
| dev.spacy  | 20% of official train data (combining Beth and Partners training data splits) | 51 |
| test.spacy | Official test data (combining Beth and Partners test data splits) | 173 |


## 🧮 Results

We've tested 4 configurations which have been saved to the `configs` folder. All models evaluated by F1 Score on the official i2b2 test data created by the `preprocess` command above at `test.spacy`.

| Model | Config | Description | F1 Score (test.spacy) |
|-------|--------|-------------|----------|
| spaCy + Vectors | `configs/spacy_config.cfg` | spaCy `ner` model with static vectors from `en_core_web_lg` | 67.54 |
| PyTorch + Vectors | `configs/config.cfg` | PyTorch based `torch_ner` pipeline with static vectors from `en_core_web_lg` | 62.69 |
| spaCy + Transformer | `configs/spacy_config_trf.cfg` |spaCy `ner` model with Transformer `roberta-base` | 75.20 |
| PyTorch + Transformer | `configs/config_trf.cfg` | PyTorch based `torch_ner` pipeline with Transformer `roberta-base` | **78.07** |

