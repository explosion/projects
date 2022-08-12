import json
import random
import tempfile
from pathlib import Path
from typing import List, Optional

import spacy
import typer
from spacy.cli._util import parse_config_overrides, setup_gpu
from spacy.cli._util import show_validation_error
from spacy.tokens import DocBin
from spacy.training.corpus import Corpus
from spacy.training.initialize import init_nlp
from spacy.training.loop import train as train_nlp
from spacy.util import load_config
from wasabi import msg

METRICS = ["token_acc", "pos_acc", "morph_acc", "tag_acc", "dep_uas", "dep_las"]


def chunk(l: List, n: int):
    """Split a list l into n chunks of fairly equal number of elements"""
    k, m = divmod(len(l), n)
    return (l[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def get_all_except(l: List, idx: int):
    """Get all elements of a list except a given index"""
    return l[:idx] + l[(idx + 1) :]


def flatten(l: List) -> List:
    """Flatten a list of lists"""
    return [item for sublist in l for item in sublist]


app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(
    # fmt: off
    ctx: typer.Context,  # this is only used to read additional arguments
    corpus_path: Path = typer.Argument(..., help="Path to the full corpus."),
    output_path: Path = typer.Argument(..., help="Path to save the output scores (JSON)."),
    config_path: Path = typer.Argument(..., help="Path to the spaCy configuration file."),
    n_folds: int = typer.Option(10, "--n-folds", "-n", help="Number of folds for cross-validation.", show_default=True),
    lang: Optional[str] = typer.Option("tl", "--lang", "-l", help="Language vocab to use.", show_default=True),
    shuffle: bool = typer.Option(False, "--shuffle", "-f", help="Flag for shuffling data"),
    use_gpu: int = typer.Option(0, help="GPU id to use. Pass -1 to use the CPU."),
    # fmt: on
):
    """Train a dependency parser with k-fold cross validation

    This command-line interface allows training a spaCy pipeline using k-fold
    cross validation. You can set the number of folds by passing a parameter to
    '--n-folds'. It performs the split automatically, so you need to pass the
    full corpus (not split into training/dev) in 'corpus_path'. Lastly,
    we get the average of the scores for each fold to obtain the final metrics.
    """
    overrides = parse_config_overrides(ctx.args)
    setup_gpu(use_gpu)

    nlp = spacy.blank(lang)
    doc_bin = DocBin().from_disk(corpus_path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    if shuffle:
        random.shuffle(docs)

    folds = list(chunk(docs, n_folds))
    all_scores = {metric: [] for metric in METRICS}
    for idx, fold in enumerate(folds):
        dev = fold
        train = flatten(get_all_except(folds, idx=idx))
        msg.divider(f"Fold {idx+1}, train: {len(train)}, dev: {len(dev)}")

        # Save the train and test dataset into a temporary directory
        # then train within the context of that directory
        with tempfile.TemporaryDirectory() as tmpdir:

            msg.info("Preparing data for training")
            overrides["paths.train"] = str(Path(tmpdir) / "tmp_train.spacy")
            overrides["paths.dev"] = str(Path(tmpdir) / "tmp_dev.spacy")
            tmp_train_docbin = DocBin(docs=train)
            tmp_train_docbin.to_disk(overrides["paths.train"])
            tmp_dev_docbin = DocBin(docs=dev)
            tmp_dev_docbin.to_disk(overrides["paths.dev"])
            msg.good(
                f"Temp files at {overrides['paths.train']} and {overrides['paths.dev']}"
            )

            msg.info("Training model for the current fold")
            with show_validation_error(config_path, hint_fill=False):
                config = load_config(config_path, overrides, interpolate=False)
                nlp = init_nlp(config)

            nlp, _ = train_nlp(nlp, None)

            msg.info("Evaluating on the dev dataset")
            corpus = Corpus(overrides["paths.dev"], gold_preproc=False)
            dev_dataset = list(corpus(nlp))
            scores = nlp.evaluate(dev_dataset)

            # For our purposes, we'll only get the scores for the morphologizer,
            # dependency parser, and POS tagger
            for metric in METRICS:
                all_scores[metric].append(scores[metric])

    msg.info(f"Computing final {n_folds}-fold cross-validation score")
    avg_scores = {
        metric: sum(scores) / len(scores) for metric, scores in all_scores.items()
    }
    msg.table(avg_scores, header=("Metric", "Score"))
    output_path.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fp:
        json.dump(avg_scores, fp, indent=4)


if __name__ == "__main__":
    app()
