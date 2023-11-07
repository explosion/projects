<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Combining Multiple Trained NER Components

This project shows you the different ways you can combine multiple trained NER components and their tradeoffs.


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
| `prep` | Install the base models. |
| `assemble` | Build two versions of the combined pipelines from configs. |
| `package` | Package the pipelines so they can be installed. |
| `check` | Use the pipeline interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `prep` &rarr; `assemble` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/ner_drugs-0.0.0.tar.gz` | URL | Pretrained drug model |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->

## Notes on Combining Components

It's possible to have more than one NER component in a single pipeline, but when
combining components there are some things you need to be aware of.

The first thing to keep in mind is that components **do not overwrite earlier
annotations**. This means that if you combine multiple components the first one
takes precedence.

Consider this example sentence:

> My name is John Benz and I remove weeds from my garden.

This sample text shows some of the effects that can have.  "Benz" is a surname,
but it also resembles the slang term "benzos" for a kind of drug.  "Weed" can
of course refer to marijuana. If the normal NER runs first it will flag "John
Benz" as a PERSON, but if the drug component is run first "Benz" will be marked
as a DRUG and "John" will be ignored.

Play around with the different configurations and see how the order affects the
annotations. You can play with the pipelines interactively using this command:

    spacy project run check

You can combine components using the configs, and that would generally be the
recommended approach, but it's also possible to do it in code. In code it looks
like this:

```python
import spacy

nlp = spacy.load("en_core_web_md") # load the base pipeline
drug_nlp = spacy.load("en_ner_drugs") # load the drug pipeline
# give this component a copy of its own tok2vec
drug_nlp.replace_listeners("tok2vec", "ner", ["model.tok2vec"])

# now you can put the drug component before or after the other ner
# This will print a W113 warning but it's safe to ignore here
nlp.add_pipe(
    "ner",
    name="ner_drug",
    source=drug_nlp,
    after="ner",
)

doc = nlp("My name is John Benz and I remove weeds from my garden.")
print(doc.ents)
# => (John Benz, weeds)
```

Another option to consider when combining NER components is what to do with
your tok2vec layer. You can either give each NER component its own tok2vec
layer, or train them together with a shared layer. 

If you've already trained the components separately, letting them keep their
own tok2vec layers will be the easiest option. The main downside is it will
increase the size of your pipeline. That's what we've done for this example since
the drug NER and core NER were trained separately.

Sharing a tok2vec layer will reduce pipeline size, but normally if you can
train the pipelines together it might be worth combining their annotations to
produce just one NER component instead. Usually combining all annotations will
work better than training separate components, so it's worth training a simple
model just to check performance, even if you don't expect it to work.

An example situation where sharing a tok2vec layer but training different
components would make sense is if your labels don't interact much, and one
group of labels has much more data than another group. If you have enough data
for all labels, but some are much more numerous, it's possible (though not
guaranteed) that they could be ignored by your model in favor of more common
labels. 

While based on a description of your data it might be possible to predict the
optimal approach, in general the best thing is to actually try a few different
configurations and use the one that performs best. You can train relatively
small models quickly and compare them, and then just scale up the architectures
once you've establish which baseline is working best.
