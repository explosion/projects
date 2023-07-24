from pathlib import Path

import typer
from input_reader import read_trial
from spacy import displacy
from spacy_llm.util import assemble
from wasabi import msg


def visualise_entities(pmid: int, config_path: Path, verbose: bool = False):
    import logging
    import spacy_llm

    spacy_llm.logger.addHandler(logging.StreamHandler())
    spacy_llm.logger.setLevel(logging.DEBUG)

    #msg.text(f"Processing PMID {pmid}", show=verbose)
    #msg.text(f"Loading config from {config_path}", show=verbose)
    text = read_trial(pmid, verbose=verbose)
    nlp = assemble(config_path)
    doc = nlp(text)
    # options = {"ents": ["Drug", "Dose"], "colors": {"Drug": "pink", "Dose": "orange"}}
    ents = list(doc.ents)
    print("ents", len(ents))
    for ent in ents:
        print(ent.text, ent.label_)
    #displacy.serve(doc, style="ent", options=options)


if __name__ == "__main__":
    typer.run(visualise_entities)
