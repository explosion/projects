# Entity Linking: disambiguation of "Emerson" mentions in sentences 

This directory contains the datasets and scripts for an example project using [spaCy](https://spacy.io)'s Entity Linking (EL) functionality 
to disambiguate "Emerson" mentions in text. First, a pretrained model is used to perform Named Entity Recognition (NER). 
Then, we create a Knowledge Base (KB) in spaCy that holds the information of the entities we want to disambiguate, 
in this example project we consider three different people called "Emerson": an Australian tennis player, 
an American writer, and a Brazilian footballer. Next, we use [Prodigy](https://prodi.gy) to create some manually 
annotated data with a custom Prodigy recipe. Finally, we create a new Entity Linking component in spaCy, 
and train it with this annotated data. We test the model on a few unseen sentences.

> 📺 **This project was created as part of a [step-by-step video tutorial](TODO LINK).**

## Scripts

All code to create the KB and the EL component in spaCy, can be found in [`el_tutorial.py`](scripts/el_tutorial.py). 
Alternatively, you can execute this code in a Jupyter notebook: [`notebook_video.ipynb`](scripts/notebook_video.ipynb). 
Both files cover the same steps:
 * Read in a predefined CSV file to define the entities in our Knowledge Base
 * Read in the manually annotated data and convert it to the right training format
 * Create a new entity linking pipe and train it
 * Apply the entity linker to some unseen data to test its performance

## Prodigy annotation

To perform the manual annotation in Prodigy, we have written a custom recipe [`el_recipe.py`](scripts/el_recipe.py).

As input, the file [`emerson_input_text.txt`](prodigy/emerson_input_text) contains 30 sentences from Wikipedia containing just 
the mention "Emerson" and not the full name. These sentences are then annotated with Prodigy by executing the command
```
prodigy entity_linker.manual emersons_annotated ../prodigy/emerson_input_text.txt ../output/my_nlp/ ../output/my_kb ../input/entitites.csv -F el_recipe.py
```
