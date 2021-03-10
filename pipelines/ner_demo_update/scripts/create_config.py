import typer
from pathlib import Path

import spacy


def create_config(model_name: str, component_to_update: str, output_path: Path):
    nlp = spacy.load(model_name)

    # create a new config as a copy of the loaded pipeline's config
    config = nlp.config.copy()

    # revert most training settings to the current defaults
    default_config = spacy.blank(nlp.lang).config
    config["corpora"] = default_config["corpora"]
    config["training"] = default_config["training"]

    # set the vectors if the loaded pipeline has vectors
    if len(nlp.vocab.vectors) > 0:
        config["paths"]["vectors"] = model_name

    # source all components from the loaded pipeline and freeze all except the
    # component to update
    config["training"]["frozen_components"] = []
    for pipe_name in nlp.component_names:
        config["components"][pipe_name] = {"source": model_name}
        if pipe_name != component_to_update:
            config["training"]["frozen_components"].append(pipe_name)

    # save the config
    config.to_disk(output_path)


if __name__ == "__main__":
    typer.run(create_config)
