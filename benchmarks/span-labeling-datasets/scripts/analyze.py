import os
import csv
import tqdm

import spacy
import typer
import pandas as pd

from wasabi import msg
from spacy.tokens import Doc, DocBin
from typing import Sequence, List, Union, Dict
from _util import SplitInfo

Number = Union[int, float]


def per_ent_stats(docs: Sequence[Doc]) -> List[Dict]:
    spans = []
    for i, doc in tqdm.tqdm(enumerate(docs), total=len(docs)):
        # Special "null span" row.
        if not doc.ents:
            row = {
                "doc_id": i,
                "text": None,
                "label": None,
                "length": None,
                "doc_length": len(doc),
                "num_ents": 0
            }
            spans.append(row)
        for span in doc.ents:
            row = {
                "doc_id": i,
                "text": span.text,
                "label": span.label_,
                "length": len(span),
                "doc_length": len(doc),
                "num_ents": len(doc.ents)
            }
            spans.append(row)
    return spans


def datastats(df: pd.DataFrame):
    doc_group = df.groupby(["doc_id"])
    msg.info(f"Num docs {len(doc_group)}")
    msg.info(f"Number of classes {df['label'].nunique()}")
    msg.info(f"Average doc-length: {doc_group['doc_length'].mean().mean()}")
    msg.info(f"Average number of entities: {doc_group['num_ents'].mean().mean()}")
    msg.info(f"Average document length: {doc_group['doc_length'].mean().mean()}")
    msg.info(f"Average entity length: {df['length'].mean()}")
    msg.info(f"Total number of entities: {len(df[df['label'].notnull()])}")


def analyze(
    docbin_path: str,
    model: str,
    *,
    data_dir: str = "corpus",
    output_dir: str = "analyses"
):
    """
    Write two .csv files one with label statistics
    and another with properties of each entity in
    the data set.
    """
    nlp = spacy.load(model)
    vocab = nlp.vocab
    docs = list(
        DocBin().from_disk(docbin_path).get_docs(vocab)
    )
    splitinfo = SplitInfo(docbin_path)
    span_stats = per_ent_stats(docs)
    df = pd.DataFrame.from_dict(span_stats)
    datastats(df)
    vocabulary = set()
    norms = set()
    prefixes = set()
    suffixes = set()
    shapes = set()
    num_tokens = 0
    for doc in docs:
        for token in doc:
            vocabulary.add(token.text)
            norms.add(token.norm_)
            prefixes.add(token.prefix_)
            suffixes.add(token.suffix_)
            shapes.add(token.shape_)
            num_tokens += 1
    vec_vocabulary = {nlp.vocab.strings[k] for k in nlp.vocab.vectors.keys()}
    msg.info(f"Vocabulary size: {len(vocabulary)}")
    msg.info(f"Unknown words: {len(vocabulary - vec_vocabulary)}")
    msg.info(f"Number of norms: {len(norms)}")
    msg.info(f"Number of prefixes: {len(prefixes)}")
    msg.info(f"Number of suffixes: {len(suffixes)}")
    msg.info(f"Number of shapes: {len(shapes)}")
    msg.info(f"Number of tokens: {num_tokens}")
    f_prefix = (f"{splitinfo.dataset}-{splitinfo.split}")
    if splitinfo.seen != "":
        f_prefix += f"-{splitinfo.seen}"
    span_stats_path = os.path.join(output_dir, f"{f_prefix}.csv")
    vocabulary_path = os.path.join(output_dir, f"{f_prefix}.vocab")
    norm_path = os.path.join(output_dir, f"{f_prefix}.norm")
    prefix_path = os.path.join(output_dir, f"{f_prefix}.prefix")
    suffix_path = os.path.join(output_dir, f"{f_prefix}.suffix")
    shape_path = os.path.join(output_dir, f"{f_prefix}.shape")
    df.to_csv(span_stats_path)
    with open(vocabulary_path, "w", encoding="utf-8") as vocabfile:
        vocabfile.write("\n".join(vocabulary))
    with open(norm_path, "w", encoding="utf-8") as normfile:
        normfile.write("\n".join(norms))
    with open(prefix_path, "w", encoding="utf-8") as prefixfile:
        prefixfile.write("\n".join(prefixes))
    with open(suffix_path, "w", encoding="utf-8") as suffixfile:
        suffixfile.write("\n".join(suffixes))
    with open(shape_path, "w", encoding="utf-8") as shapefile:
        shapefile.write("\n".join(shapes))


if __name__ == "__main__":
    typer.run(analyze)
