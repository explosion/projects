<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Combining Multiple Trained NER Components

This project shows you the different ways you can combine multiple trained NER components and their tradeoffs.
Note that before running this project, you should run the [ner_drugs](../ner_drugs) project and install the packaged version of it.


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
| `assemble` | Build two versions of the combined model from configs. |
| `package` | Package the models so they can be installed. |
| `visualize-model` | Use the model interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `assemble` |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

## Notes on Combining Models

It's possible to have more than one NER model in a single pipeline, but when
combining models there are some things you need to be aware of.

The first thing to keep in mind about models is that models **do not overwrite
earlier annotations**. This means that if you combine multiple models the first
one takes precedence.

The sample text for this model shows and example of the effects that can have.
"Benz" is a surname, but it also resembles the slang term "benzos" for a kind
of drug. "weed" can of course refer to marijuana. If the normal NER model runs
first it will flag "John Benz" as a PERSON, but if the drug model is run first
"Benz" will be marked as a DRUG and "John" will be ignored.

Play around with the different models and see how the order affects the
annotations. 

You can combine models using the configs, but it's also possible to do it in
code. In code it looks like this:

```python
import spacy

nlp = spacy.load("en_core_web_md") # load the base model
drug = spacy.load("en_ner_drugs") # load the drug model

# now you can put the drug model before or after the other ner
nlp.add_pipe(
    "ner",
    name="ner_drug",
    source=drug_nlp,
    after="ner",
    #TODO this doesn't actually work - weights are wrong?
    config={"replace_listeners": ["model.tok2vec"]},
)
```

Another option to consider when combining NER components is what to do with
your tok2vec layer. You can either give each NER component its own tok2vec
layer, or train them together with a shared layer. 

If you've already trained the components separately, letting them keep their
own tok2vec layers will be the easiest option. The main downside is it will
increase the size of your model. That's what we've done for this example since
the drug NER and core NER were trained separately.

Sharing a tok2vec layer will reduce model size, but normally if you can train
the models together it might be worth combining their annotations to produce
just one NER component instead. Usually combining all annotations will work
better than training separate models, so it's worth training a simple model
just to check performance, even if you don't expect it to work.

An example situation where sharing a tok2vec layer but training different
models would make sense is if your labels don't interact much, and one group of
labels has much more data than another group. If you have enough data for all
labels, but some are much more numerous, it's possible (though not guaranteed)
that they could be ignored by your model in favor of more common labels. 

While based on a description of your data it might be possible to predict the
optimal approach, in general the best thing is to actually try a few different
configurations and use the one that performs best. You can train relatively
small models quickly and compare them, and then just scale up the architectures
once you've establish which baseline is working best.
