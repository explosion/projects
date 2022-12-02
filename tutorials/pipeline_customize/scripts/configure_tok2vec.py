from pathlib import Path

import spacy
import typer
from util import check_tok2vecs, get_tok2vecs, get_listeners

def use_transformer(
    base_model: str, output_path: Path, transformer_name: str = "roberta-base"
):
    """Replace pipeline tok2vec with transformer, update listeners, output config."""
    # 1. identify tok2vec
    # 2. replace tok2vec
    # 3. replace listeners
    nlp = spacy.load(base_model)
    check_tok2vecs(base_model, nlp.config)

    tok2vecs = get_tok2vecs(nlp.config)
    assert len(tok2vecs) > 0, "Must have tok2vec to replace!"

    nlp.remove_pipe(tok2vecs[0])
    # the rest can be default values
    trf_config = {
        "model": {
            "name": transformer_name,
        }
    }
    trf = nlp.add_pipe("transformer", config=trf_config, first=True)

    # TODO maybe remove vectors?

    # now update the listeners
    listeners = get_listeners(nlp)
    for listener in listeners:
        listener_config = {
            "@architectures": "spacy-transformers.TransformerListener.v1",
            "grad_factor": 1.0,
            "upstream": "transformer",
            "pooling": {"@layers": "reduce_mean.v1"},
        }
        nlp.config["components"][listener]["model"]["tok2vec"] = listener_config

    # that's it!
    nlp.config.to_disk(output_path)


def use_tok2vec(base_model: str, output_path: Path):
    """Replace pipeline tok2vec with CNN tok2vec, update listeners, output config."""
    nlp = spacy.load(base_model)
    check_tok2vecs(base_model, nlp.config)

    tok2vecs = get_tok2vecs(nlp.config)
    assert len(tok2vecs) > 0, "Must have tok2vec to replace!"

    nlp.remove_pipe(tok2vecs[0])

    tok2vec = nlp.add_pipe("tok2vec", first=True)
    width = "${components.tok2vec.model.encode:width}"

    listeners = get_listeners(nlp)
    for listener in listeners:
        listener_config = {
            "@architectures": "spacy.Tok2VecListener.v1",
            "width": width,
            "upstream": "tok2vec",
        }
        nlp.config["components"][listener]["model"]["tok2vec"] = listener_config

    nlp.config.to_disk(output_path)


if __name__ == "__main__":

    help_msg = """
    This script will help you swap out a tok2vec in your pipeline for a
    Transformer or vice-versa.
    """
    app = typer.Typer(name="tok2vec Swapper", help=help_msg, no_args_is_help=True)
    app.command("use-transformer")(use_transformer)
    app.command("use-tok2vec")(use_tok2vec)
    app()
