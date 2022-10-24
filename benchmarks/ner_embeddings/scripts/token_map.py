import srsly
import typer
from wasabi import msg
from tqdm import tqdm

from typing import Optional, List, Dict
from collections import Counter
from pathlib import Path

from thinc.api import ConfigValidationError
from spacy.attrs import intify_attr
from spacy.tokens import Doc
from spacy import util
from spacy.schemas import ConfigSchemaTraining
from spacy.cli._util import Arg, Opt, show_validation_error
from spacy.cli._util import parse_config_overrides, import_code
from spacy.errors import Errors

app = typer.Typer()


@app.command(
    context_settings={
        "allow_extra_args": True, "ignore_unknown_options": True
    },
)
def init_tables_cli(
    ctx: typer.Context,
    config_path: Path = Arg(..., help="Path to config file", exists=True, allow_dash=True),
    output_path: Path = Arg(..., help="Output directory for the mappers"),
    code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    unk: int = Opt(0, help="id of the 'unknown symbol' for all tables"),
    limit: int = Opt(0, help="Number of documents to run through."),
    min_freq: int = Opt(
        0, help="Minimum number of times a symbol has to occur to include"
        )
) -> None:
    if not output_path.exists():
        output_path.mkdir(parents=True)
    overrides = parse_config_overrides(ctx.args)
    with show_validation_error(config_path):
        config = util.load_config(
            config_path, overrides=overrides, interpolate=True
        )
    embedder = config["components"]["tok2vec"]["model"]["embed"]
    if embedder["@architectures"] != "spacy.MultiEmbed.v1":
        raise ValueError(
            "Can only run init tables command for pipeline with "
            "a spacy.MultiEmbed.v1 component."
        )
    import_code(code_path)
    nlp = util.load_model_from_config(config, auto_fill=True)
    msg.good("Succesfully loaded pipeline.")
    training = util.registry.resolve(
        config["training"], schema=ConfigSchemaTraining
    )
    train_corpus = training["train_corpus"]
    if not isinstance(train_corpus, str):
        raise ConfigValidationError(
            desc=Errors.E897.format(
                field="training.train_corpus",
                type=type(training["train_corpus"])
            )
        )
    train_corpus = util.resolve_dot_names(config, [train_corpus])[0]
    attrs = embedder["attrs"]
    unk = embedder["unk"]
    if not isinstance(attrs, list):
        raise ValueError(
            "The 'attrs' field in `MultiEmbed` has to be provided "
            "as a List[str]."
        )
    elif not all([isinstance(x, str) for x in attrs]):
        raise ValueError(
            "The 'attrs' field in 'MultiEmbed' has to be provided "
            "as a List[str]"
        )
    elif not isinstance(unk, int):
        raise ValueError(
            "'unk' has to be an integer, but found ({type(unk)}"
        )
    msg.text("Loading training documents.")
    train_docs = []
    if limit == 0:
        limit = float("inf")
    for i, doc in tqdm(enumerate(train_corpus(nlp))):
        if i < limit:
            train_docs.append(doc.predicted)
    msg.good(f"Loaded {len(train_docs)} documents.")
    attrs_counts, mappers = _init_tables(attrs, train_docs, unk, min_freq)
    output_stem = str(output_path / train_corpus.path.stem)
    tables_path = output_stem + ".tables"
    counts_path = output_stem + ".counts"
    srsly.write_msgpack(tables_path, mappers)
    msg.good(f"Tables saved to {tables_path}.")
    srsly.write_msgpack(output_stem + ".counts", attrs_counts)
    msg.good(f"Attribute counts saved to {counts_path}.")


def _init_tables(
        attrs: List[str],
        train_docs: List[Doc],
        unk: int,
        min_freq: Optional[int] = 0
):
    attrs_counts = {}
    msg.text("Counting attributes.")
    for attr in attrs:
        attr_id = intify_attr(attr)
        counts = Counter()
        for doc in tqdm(train_docs):
            counts.update(doc.count_by(attr_id))
        attrs_counts[attr] = counts
    # Create mappers
    mappers: Dict[str, Dict[int, int]] = {}
    msg.text("Creating mappers.")
    for i, attr in enumerate(attrs):
        sorted_counts = attrs_counts[attr].most_common()
        mappers[attr] = {}
        new_id = 0
        for symbol, count in sorted_counts:
            if min_freq:
                if count < min_freq:
                    break
            # Leave the id for the unknown symbol out of the mapper.
            if new_id == unk:
                new_id += 1
            mappers[attr][symbol] = new_id
            new_id += 1
        msg.info(
            f"Storing {len(mappers[attr])} entries for {attr}."
        )
    return attrs_counts, mappers


if __name__ == "__main__":
    app()
