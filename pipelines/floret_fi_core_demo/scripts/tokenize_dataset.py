import re
import spacy
import typer
from itertools import islice
from pathlib import Path
from datasets import load_dataset


def main(
    lang: str,
    oscar_dataset: str,
    max_texts: int,
    output_file: Path,
    n_process: int = 8,
    batch_size: int = 100,
):
    if lang == "ko":
        nlp = spacy.blank(
            "ko", config={"nlp": {"tokenizer": {"@tokenizers": "spacy.Tokenizer.v1"}}}
        )
    elif lang == "zh":
        nlp = spacy.blank("zh", config={"nlp": {"tokenizer": {"segmenter": "pkuseg"}}})
        nlp.tokenizer.initialize(pkuseg_model="spacy_ontonotes")
    else:
        nlp = spacy.blank(lang)

    nlp.add_pipe("sentencizer")
    nlp.max_length = 10 ** 8

    dataset = load_dataset("oscar", oscar_dataset, split="train", streaming=True)
    with open(output_file, "w", encoding="utf8") as output_fileh:
        texts = (
            re.sub("\s+", " ", line["text"].strip())
            for line in islice(iter(dataset), max_texts)
        )
        for doc in nlp.pipe(texts, n_process=n_process, batch_size=batch_size):
            for sent in doc.sents:
                output_fileh.write(" ".join([t.text for t in sent]) + "\n")


if __name__ == "__main__":
    typer.run(main)
