import tqdm
import typer
import spacy
import srsly

import numpy as np
import pandas as pd

from collections import Counter
from typing import List, Hashable, Dict
from pathlib import Path

from wasabi import msg
from spacy.util import ensure_path
from spacy.tokens import DocBin
from spacy.attrs import intify_attr
from thinc.backends.numpy_ops import NumpyOps


OPS = NumpyOps()


def _attr_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    How many unique symbols are in the df per attribute.
    """
    return dict(df['attr'].value_counts())


def _unused(df: pd.DataFrame, attr: str) -> int:
    """
    Return the number of unused buckets.
    """
    n = df.attrs['n_rows'][attr]
    df = df.loc[df["attr"] == attr]
    hashrows = df.to_numpy()[:, [1, 2, 3, 4]]
    hashrows = hashrows.astype(int)
    unique_keys = np.unique(hashrows)
    return n - len(unique_keys)


def _collisions(df, attr, k=20):
    """
    Number of times one of the top-k most frequent symbols
    in category attr shares 1, 2, 3 or 4 hashes with any other
    symbol.
    """
    df = df.loc[df["attr"] == attr]
    k = min(k, len(df))
    sorted_df = df.sort_values("count", ascending=False)
    hashrows = sorted_df.to_numpy()[:, [1, 2, 3, 4]]
    hashrows = hashrows.astype(int)
    hashrows = list(map(set, list(hashrows)))
    collides = np.zeros((k, 5))
    for i in tqdm.tqdm(range(0, k)):
        for j in range(0, len(hashrows)):
            if i == j:
                continue
            n_overlap = len(hashrows[i].intersection(hashrows[j]))
            collides[i, n_overlap] += 1
    return collides


def make_symbol_stats(
    data_path: Path,
    output_path: Path,
    lang: str,
    attrs: List[str] = typer.Option(["PREFIX", "SUFFIX", "SHAPE", "NORM"]),
    rows: List[int] = typer.Option([2500, 2500, 2500, 5000]),
    seed: int = typer.Option(42),
    k: int = typer.Option(100)
):
    """
    Go through all documents in a .spacy file and store two
    pieces of information about each encountered token (feature):
    1.) number if times it appears across all documents
    2.) the four keys it gets hashed to in the HashEmbed table.

    """
    if len(attrs) != len(rows):
        raise ValueError(
            "Number of attributes has to be the same "
            "as the number of rows."
        )
    data_path = ensure_path(data_path)
    output_path = ensure_path(output_path)
    if not output_path.is_dir():
        raise ValueError(
            "Output path must be a directory."
        )
    nlp = spacy.blank(lang)
    docs = DocBin().from_disk(data_path).get_docs(nlp.vocab)
    docs = list(docs)
    msg.good(f"Loaded data {data_path}")
    msg.good(f"Checking collisions with top-{k} tokens")
    data = {}
    for attr, n_row in zip(attrs, rows):
        counts = Counter()
        hashes = {}
        seen = set()
        for doc in docs:
            doc_counts = doc.count_by(intify_attr(attr))
            counts.update(doc_counts)
            seen.update(list(doc_counts.keys()))
            symbol_idx = np.array(list(doc_counts.keys()), dtype="uint64")
            doc_hashes = OPS.hash(symbol_idx, seed) % n_row
            for symbol, keys in zip(symbol_idx, doc_hashes):
                # Same symbol should always be mapped to the same keys
                if symbol in hashes:
                    assert hashes[symbol] == tuple(keys)
                else:
                    hashes[symbol] = tuple(keys)
        for symbol, count in counts.items():
            datakey = attr + str(symbol)
            assert datakey not in data
            data[datakey] = [count, *hashes[symbol], attr]
    df = pd.DataFrame.from_dict(data, orient="index")
    df.columns = ['count', 'hash1', 'hash2', 'hash3', 'hash4', 'attr']
    df.attrs['n_rows'] = {attr: n_rows for attr, n_rows in zip(attrs, rows)}
    results = {}
    results["counts"] = _attr_counts(df)
    for attr in attrs:
        results[attr] = {}
        results[attr]["unused"] = _unused(df, attr)
        collisions = _collisions(df, attr, k=k)
        results[attr]["collisions"] = dict(enumerate(list(collisions.mean(0))))
        del results[attr]["collisions"][0]
    df.to_csv(output_path / f"{data_path.name}.csv")
    srsly.write_msgpack(
        output_path / f"{data_path.name}.msg", results
    )


if __name__ == "__main__":
    typer.run(make_symbol_stats)
