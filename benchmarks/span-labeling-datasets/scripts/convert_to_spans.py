"""Monkey-patched version of the convert command that transfers entities to Doc.spans"""

import random
from pathlib import Path
from typing import Iterable, Optional, Union

import srsly
import typer
from spacy.cli.convert import CONVERTERS, _get_converter, _write_docs_to_file
from spacy.cli.convert import verify_cli_args, walk_directory
from spacy.tokens import Doc, DocBin, SpanGroup
from wasabi import Printer

FILE_TYPE = "spacy"
Arg = typer.Argument
Opt = typer.Option


def convert_cli(
    # fmt: off
    input_path: Path = Arg(..., help="Input file or directory", exists=True),
    output_dir: Path = Arg("-", help="Output directory. '-' for stdout.", allow_dash=True, exists=True),
    spans_key: str = Opt("sc", "--spans-key", "-sc", help="Spans key to use when storing entities"),
    n_sents: int = Opt(1, "--n-sents", "-n", help="Number of sentences per doc (0 to disable)"),
    seg_sents: bool = Opt(False, "--seg-sents", "-s", help="Segment sentences (for -c ner)"),
    model: Optional[str] = Opt(None, "--model", "--base", "-b", help="Trained spaCy pipeline for sentence segmentation to use as base (for --seg-sents)"),
    morphology: bool = Opt(False, "--morphology", "-m", help="Enable appending morphology to tags"),
    merge_subtokens: bool = Opt(False, "--merge-subtokens", "-T", help="Merge CoNLL-U subtokens"),
    converter: str = Opt("auto", "--converter", "-c", help=f"Converter: {tuple(CONVERTERS.keys())}"),
    ner_map: Optional[Path] = Opt(None, "--ner-map", "-nm", help="NER tag mapping (as JSON-encoded dict of entity types)", exists=True),
    lang: Optional[str] = Opt(None, "--lang", "-l", help="Language (if tokenizer required)"),
    use_ents: bool = Opt(False, "--use-ents", "-e", help="Use Doc.ents, don't transfer to Doc.spans"),
    train_size: Optional[float] = Opt(None, "--train-size", "-sz", help="Size of the training dataset for splitting"),
    shuffle: bool = Opt(False, "--shuffle", "-sf", help="Shuffle the dataset before splitting"),
    seed: Optional[int] = Opt(None, "--seed", "-sd", help="Random seed for shuffling the data")
    # fmt: on
):
    """
    Convert files into json or DocBin format for training. The resulting .spacy
    file can be used with the train command and other experiment management
    functions.

    If no output_dir is specified and the output format is JSON, the data
    is written to stdout, so you can pipe them forward to a JSON file:
    $ spacy convert some_file.conllu --file-type json > some_file.json

    NOTE: This is a monkeypatched version of the original `convert` command.
    Here, we added an additional step that transfers the entitites into
    the Doc.spans attribute for Span Categorization.

    DOCS: https://spacy.io/api/cli#convert
    """
    input_path = Path(input_path)
    output_dir: Union[str, Path] = "-" if output_dir == Path("-") else output_dir
    silent = output_dir == "-"
    msg = Printer(no_print=silent)
    verify_cli_args(msg, input_path, output_dir, FILE_TYPE, converter, ner_map)
    converter = _get_converter(msg, converter, input_path)
    convert(
        input_path,
        output_dir,
        n_sents=n_sents,
        seg_sents=seg_sents,
        model=model,
        morphology=morphology,
        merge_subtokens=merge_subtokens,
        converter=converter,
        ner_map=ner_map,
        lang=lang,
        silent=silent,
        msg=msg,
        spans_key=spans_key,
        use_ents=use_ents,
        train_size=train_size,
        shuffle=shuffle,
        seed=seed,
    )


def transfer_ents_to_spans(docs: Iterable[Doc], spans_key: str) -> Iterable[Doc]:
    _docs = []
    for doc in docs:
        spans = [ent for ent in doc.ents]
        group = SpanGroup(doc, name=spans_key, spans=spans)
        doc.spans[spans_key] = group
        doc.set_ents([])
        _docs.append(doc)
    return _docs


def _save_docs_to_disk(
    docs: Iterable[Doc],
    output_dir: Union[str, Path],
    input_loc: Path,
    is_dev: bool,
    msg: Printer,
):
    db = DocBin(docs=docs, store_user_data=True)
    len_docs = len(db)
    data = db.to_bytes()  # type: ignore[assignment]

    if is_dev:
        filename = input_loc.stem + "-dev" + input_loc.suffix
    else:
        filename = input_loc.parts[-1]

    output_file = Path(output_dir) / filename
    output_file = output_file.with_suffix(f".{FILE_TYPE}")
    _write_docs_to_file(data, output_file, FILE_TYPE)
    msg.good(f"Generated output file ({len_docs} documents): {output_file}")


def convert(
    input_path: Path,
    output_dir: Union[str, Path],
    *,
    n_sents: int = 1,
    seg_sents: bool = False,
    model: Optional[str] = None,
    morphology: bool = False,
    merge_subtokens: bool = False,
    converter: str = "auto",
    ner_map: Optional[Path] = None,
    lang: Optional[str] = None,
    silent: bool = True,
    msg: Optional[Printer] = None,
    spans_key: str = "sc",
    use_ents: bool = False,
    train_size: Optional[float] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> None:
    input_path = Path(input_path)
    if not msg:
        msg = Printer(no_print=silent)
    ner_map = srsly.read_json(ner_map) if ner_map is not None else None
    for input_loc in walk_directory(input_path, converter):
        with input_loc.open("r", encoding="utf-8") as infile:
            input_data = infile.read()
        # Use converter function to convert data
        func = CONVERTERS[converter]
        docs = func(
            input_data,
            n_sents=n_sents,
            seg_sents=seg_sents,
            append_morphology=morphology,
            merge_subtokens=merge_subtokens,
            lang=lang,
            model=model,
            no_print=silent,
            ner_map=ner_map,
        )
        docs = list(docs)
        # Monkeypatched version converting docs to spans
        if not use_ents:
            msg.info("Transferring entities to doc.spans")
            docs = transfer_ents_to_spans(docs, spans_key)

        if train_size:
            msg.info(f"Splitting files with train_size {train_size}")
            if shuffle:
                if seed:
                    msg.info(f"Using random seed {seed}")
                    random.seed(seed)
                msg.info("Shuffling the documents before splitting")
                random.shuffle(docs)
            num_training = int(train_size * len(docs))
            train_docs = docs[:num_training]
            dev_docs = docs[num_training:]
            msg.text(
                f"Dataset has been split with train size={len(train_docs)} "
                f"and dev size={len(dev_docs)}"
            )

            _save_docs_to_disk(train_docs, output_dir, input_loc, is_dev=False, msg=msg)
            _save_docs_to_disk(dev_docs, output_dir, input_loc, is_dev=True, msg=msg)
        else:
            _save_docs_to_disk(docs, output_dir, input_loc, is_dev=False, msg=msg)


if __name__ == "__main__":
    typer.run(convert_cli)
