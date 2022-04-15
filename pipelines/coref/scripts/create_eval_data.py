#!/usr/bin/env python3
# Read in a file and produce evaluation data using a model.
# Evaluation data will be a `.response` and `.key` file.

import os
import sys
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import spacy
from spacy.tokens import DocBin

PREFIX = "coref_clusters"


def write_key(fname, doc, name):
    # save the list of annotations for each token
    tok2cluster = defaultdict(list)
    for key, cluster in doc.spans.items():
        if not key.startswith(PREFIX):
            continue

        # span key looks like "coref_clusters_0"
        cluster_id = key.split("_")[-1]

        for mention in cluster:
            for tok in mention:
                # write the mention key
                if len(mention) == 1:
                    mkey = f"({cluster_id})"
                elif tok == mention[0]:
                    mkey = f"({cluster_id}"
                elif tok == mention[-1]:
                    mkey = f"{cluster_id})"
                else:
                    mkey = str(cluster_id)

                tok2cluster[tok.i].append(mkey)

    # now write out the data we have
    out = f"#begin document ({name})\n"

    for sent in doc.sents:
        # separate sentence with blank line, except the first
        if sent[0].i > 0:
            out += "\n"
        for tok in sent:
            tok_id_in_sent = str(tok.i - sent[0].i)

            coref_data = "|".join(tok2cluster[tok.i])
            if not coref_data:
                coref_data = "-"
            # first two values are ignored but traditionally are:
            # - doc id
            # - part number (document sub-id)
            out += "\t".join(["_", "0", tok_id_in_sent, tok.text, coref_data]) + "\n"
    out += "#end document\n"
    with open(fname, "a") as ofile:
        ofile.write(out)


def main():
    model_name = sys.argv[1]
    infile = Path(sys.argv[2])
    # output
    response = infile.with_suffix(".response")
    key = infile.with_suffix(".key")

    nlp = spacy.load(model_name)
    db = DocBin().from_disk(infile)
    docs = db.get_docs(nlp.vocab)
    # progress
    docs = tqdm(docs, total=len(db))

    for ii, doc in enumerate(docs):
        if len(doc) == 0:
            print("WARNING: empty doc")
            continue
        # produce the key data
        write_key(key, doc, f"test{ii}")
        # strip gold span data
        skeys = list(doc.spans.keys())
        for skey in skeys:
            del doc.spans[skey]
        # predict
        doc = nlp(doc)
        # produce the response (prediction) data
        write_key(response, doc, f"test{ii}")


if __name__ == "__main__":
    main()
