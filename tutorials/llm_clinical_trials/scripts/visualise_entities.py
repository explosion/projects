import logging
from pathlib import Path

import spacy_llm
import typer
from input_reader import read_trial
from spacy import displacy
from spacy_llm.util import assemble
from wasabi import msg

DEBUG = False
PRINT_CONSOLE = True
PRINT_DISPLACY = False


def visualise_entities(pmid: int, config_path: Path, verbose: bool = False):
    spacy_llm.logger.addHandler(logging.StreamHandler())
    if DEBUG:
        spacy_llm.logger.setLevel(logging.DEBUG)

    msg.info(f"Processing PMID {pmid}", show=verbose)
    msg.info(f"Loading config from {config_path}", show=verbose)
    text = read_trial(pmid, verbose=verbose)
    nlp = assemble(config_path)
    doc = nlp(text)
    ents = list(doc.ents)
    if PRINT_CONSOLE:
        msg.text(f" - Number of entities: {len(ents)}")
        for ent in ents:
            msg.text(f"    - {ent.text} [{ent.label_}]")
    if PRINT_DISPLACY:
        options = {
            "ents": ["Drug", "Dose"],
            "colors": {"Drug": "pink", "Dose": "orange"},
        }
        displacy.serve(doc, style="ent", options=options)


if __name__ == "__main__":
    typer.run(visualise_entities)
