from pathlib import Path

import typer
from input_reader import read_trial
from spacy_llm.util import assemble
from trial_task import make_trial_task
from wasabi import msg


def run_pipeline(pmid: int, config_path: Path, verbose: bool = False):
    msg.text(f"Processing PMID {pmid}", show=verbose)
    msg.text(f"Loading config from {config_path}", show=verbose)
    text = read_trial(pmid, verbose=verbose)
    nlp = assemble(config_path)
    doc = nlp(text)

    print(doc._.trial_summary)
    print()
    for ent in doc.ents:
        print(ent.label_, ent.text)


if __name__ == "__main__":
    typer.run(run_pipeline)
