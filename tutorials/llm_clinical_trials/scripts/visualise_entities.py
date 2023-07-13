from pathlib import Path

import typer
from input_reader import read_trial
from spacy import displacy
from spacy_llm.util import assemble
from wasabi import msg


def visualise_entities(pmid: int, config_path: Path, verbose: bool = False):
    msg.text(f"Processing PMID {pmid}", show=verbose)
    msg.text(f"Loading config from {config_path}", show=verbose)
    text = read_trial(pmid, verbose=verbose)
    nlp = assemble(config_path)
    doc = nlp(text)
    options = {"ents": ["Drug", "Dose"], "colors": {"Drug": "pink", "Dose": "orange"}}
    displacy.serve(doc, style="ent", options=options)


if __name__ == "__main__":
    typer.run(visualise_entities)
