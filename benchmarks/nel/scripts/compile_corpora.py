import os.path
import pickle
from typing import List
import yaml
import typer
from pathlib import Path
from spacy.tokens import DocBin
from spacy.tokens.doc import Doc
import numpy
import reddit


def serialize_corpora(
    docs: List[Doc], frac_train: float, frac_dev: float, frac_test: float, dataset_id: str
) -> None:
    """ Serializes corpora.
    docs (List[Doc]): List of documents with entity annotations.
    frac_train (float): Fraction of documents in training set.
    frac_dev (float): Fraction of documents in dev set.
    frac_eval (float): Fraction of documents in eval set.
    dataset_id (str): Dataset ID to use as suffix in corpora filenames.
    """

    assert frac_train + frac_dev + frac_test == 1

    indices = {
        dataset: idx
        for dataset, idx in zip(
            ("train", "dev", "test"),
            numpy.split(
                numpy.asarray(range(len(docs))), [int(frac_train * len(docs)), int((frac_train + frac_dev) * len(docs))]
            )
        )
    }

    corpora_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "corpora"
    for key in indices:
        corpus = DocBin(store_user_data=True)
        for idx in indices[key]:
            corpus.add(docs[idx])
        if not os.path.exists(corpora_root / dataset_id):
            os.mkdir(corpora_root / dataset_id)
        corpus.to_disk(corpora_root / dataset_id / f"{key}_.spacy")


def main(dataset_id: str, nlp_dir: Path, dataset_config_path: Path):
    """ Create corpora in spaCy format.
    dataset_id (str): Dataset ID.
    nlp_dir (str): Directory with serialized NLP data.
    dataset_config_path (Path): Path to corpus config file.
    """

    assert dataset_id in ("reddit",)

    # Load entity info and corpus config.
    asset_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "assets"
    with open(asset_root / dataset_id / "entities.pkl", "rb") as file:
        entities_info = pickle.load(file)
    with open(asset_root / dataset_id / "entities_failed_lookups.pkl", "rb") as file:
        entities_failed_lookups = pickle.load(file)
    with open(asset_root / dataset_id / "annotations.pkl", "rb") as file:
        annotations = pickle.load(file)
    with open(dataset_config_path, "r") as stream:
        corpus_config = yaml.safe_load(stream)

    serialize_corpora(
        docs=reddit.create_corpus(
            nlp_dir / f"{dataset_id}.nlp",
            asset_root / dataset_id,
            entities_info,
            entities_failed_lookups,
            annotations,
            corpus_config["reddit"]
        ),
        frac_train=corpus_config["reddit"]["frac_train"],
        frac_dev=corpus_config["reddit"]["frac_test"],
        frac_test=corpus_config["reddit"]["frac_test"],
        dataset_id="reddit"
    )


if __name__ == "__main__":
    # typer.run(main)
    main(
        "reddit",
        Path("/home/raphael/dev/spacy-projects/benchmarks/nel/temp"),
        Path("/home/raphael/dev/spacy-projects/benchmarks/nel/configs/datasets.yml"),
    )
