import typer
import srsly
from pathlib import Path
from spacy.tokens import DocBin
import spacy


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = typer.Argument(..., dir_okay=False),
):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    data_tuples = ((eg["text"], eg) for eg in srsly.read_jsonl(input_path))
    for doc, eg in nlp.pipe(data_tuples, as_tuples=True):
        if eg["answer"] == "ignore":
            continue
        doc.cats[eg["label"]] = eg["answer"] == "accept"
        doc_bin.add(doc)
    doc_bin.to_disk(output_path)
    print(f"Processed {len(doc_bin)} documents: {output_path.name}")


if __name__ == "__main__":
    typer.run(main)
