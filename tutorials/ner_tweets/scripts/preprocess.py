from pathlib import Path

import spacy
import srsly
import typer
from spacy.tokens import Doc, DocBin
from spacy.util import get_words_and_spaces
from tqdm import tqdm
from wasabi import msg


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = typer.Argument(..., dir_okay=False),
    text_only: bool = False,
):
    """Preprocess JSONL files into the .spacy filetype

    input_path (Path): path to the JSONL file
    output_path (Path): output path once saved to the spaCy format
    text_only (bool): if True, then just saves text without other info
    """
    nlp = spacy.blank("en")
    doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])  # we're just concerned with NER

    for record in tqdm(srsly.read_jsonl(input_path)):
        if text_only:
            doc = nlp(record["text"])
        else:
            if record["answer"] != "accept":
                continue
            tokens = [token["text"] for token in record["tokens"]]
            words, spaces = get_words_and_spaces(tokens, record["text"])
            doc = Doc(nlp.vocab, words=words, spaces=spaces)
            doc.ents = [
                doc.char_span(s["start"], s["end"], label=s["label"])
                for s in record.get("spans", [])
            ]
        doc_bin.add(doc)
    doc_bin.to_disk(output_path)
    msg.good(f"Processed {len(doc_bin)} documents: {output_path.name}")


if __name__ == "__main__":
    typer.run(main)
