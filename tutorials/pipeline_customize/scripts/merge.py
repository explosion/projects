from pathlib import Path

from util import get_tok2vecs, check_tok2vecs, has_listener
from util import check_pipeline_names, get_listeners

import typer
import spacy
from spacy.language import Language
from wasabi import msg


def inner_merge(nlp, nlp2, replace_listeners=False) -> Language:
    """Actually do the merge.

    nlp: Base pipeline to add components to.
    nlp2: Pipeline to add components from.
    replace_listeners (bool): Whether to replace listeners. Usually only true
      if there's one listener.
    returns: assembled pipeline.
    """

    # we checked earlier, so there's definitely just one
    tok2vec_name = get_tok2vecs(nlp2.config)[0]
    rename = check_pipeline_names(nlp, nlp2)

    if len(get_listeners(nlp2)) > 1:
        if replace_listeners:
            msg.warn(
                """
                Replacing listeners for multiple components. Note this can make
                your pipeline large and slow. Consider chaining pipelines (like
                nlp2(nlp(text))) instead.
                """
            )
        else:
            # TODO provide a guide for what to do here
            msg.warn(
                """
                The result of this merge will have two feature sources
                (tok2vecs) and multiple listeners. This will work for
                inference, but will probably not work when training without
                extra adjustment. If you continue to train the pipelines
                separately this is not a problem.
                """
            )

    for comp in nlp2.pipe_names:
        if replace_listeners and comp == tok2vec_name:
            # the tok2vec should not be copied over
            continue
        if replace_listeners and has_listener(nlp2, comp):
            # TODO does "model.tok2vec" work for everything?
            nlp2.replace_listeners(tok2vec_name, comp, ["model.tok2vec"])
        nlp.add_pipe(comp, source=nlp2, name=rename.get(comp, comp))
        if comp in rename:
            msg.info(f"Renaming {comp} to {rename[comp]} to avoid collision...")
    return nlp


def merge_pipelines(base_model: str, added_model: str, output_path: Path):
    """Combine components from multiple pipelines into a single new one."""
    nlp = spacy.load(base_model)
    nlp2 = spacy.load(added_model)

    # to merge models:
    # - lang must be the same
    # - vectors must be the same
    # - vocabs must be the same (how to check?)
    # - tokenizer must be the same (only partially checkable)
    if nlp.lang != nlp2.lang:
        msg.fail("Can't merge - languages don't match", exits=1)

    # check vector equality
    if (
        nlp.vocab.vectors.shape != nlp2.vocab.vectors.shape
        or nlp.vocab.vectors.key2row != nlp2.vocab.vectors.key2row
        or nlp.vocab.vectors.to_bytes(exclude=["strings"])
        != nlp2.vocab.vectors.to_bytes(exclude=["strings"])
    ):
        msg.fail("Can't merge - vectors don't match", exits=1)

    if nlp.config["nlp"]["tokenizer"] != nlp2.config["nlp"]["tokenizer"]:
        msg.fail("Can't merge - tokenizers don't match", exits=1)

    # Check that each pipeline only has one feature source
    check_tok2vecs(base_model, nlp.config)
    check_tok2vecs(added_model, nlp2.config)

    # Check how many listeners there are and replace based on that
    # TODO: option to recognize frozen tok2vecs
    # TODO: take list of pipe names to copy
    listeners = get_listeners(nlp2)
    replace_listeners = len(listeners) == 1
    print(replace_listeners, len(listeners))
    nlp_out = inner_merge(nlp, nlp2, replace_listeners=replace_listeners)

    # write the final pipeline
    nlp.to_disk(output_path)
    msg.info(f"Saved pipeline to: {output_path}")


if __name__ == "__main__":

    app = typer.Typer(name="Pipeline Merge Helper")
    app.command("merge", no_args_is_help=True)(merge_pipelines)
    app()
