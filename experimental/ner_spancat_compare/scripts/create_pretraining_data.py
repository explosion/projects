import json
from pathlib import Path

from tqdm import tqdm
from wasabi import msg
import typer


def main(
    documents_path: Path = typer.Argument(
        ..., exists=True, dir_okay=True, help="Path to the documents directory"
    ),
    output_path: Path = typer.Argument(
        ..., exists=True, dir_okay=True, help="Path to the pretraining directory"
    ),
):

    # Get all file names inside /documents/ with .text extension
    p = documents_path.glob("*.text")
    files = [x for x in p if x.is_file()]
    json_file = []
    max_length = 0
    min_length = 0

    replace_dict = {"[": "", "]": "", "'": '"', "\n": " "}

    # Read all files
    msg.info(f"Found {len(files)} documents")
    for file in tqdm(files, total=len(files)):
        with open(file, "r", encoding="utf-8") as reader:
            text = str(reader.read())

            for replace in replace_dict:
                text = text.replace(replace, replace_dict[replace])

            json_line = {"text": text.strip()}
            json_file.append(json_line)

            # Getting KPI's
            if len(text) > max_length:
                max_length = len(text)
            elif len(text) < min_length:
                min_length = len(text)
            elif min_length == 0:
                min_length = len(text)

    # Write text as .jsonl
    with open(output_path / "pretraining_data.jsonl", "w", encoding="utf8") as writer:
        for entry in json_file:
            json.dump(entry, writer)
            writer.write("\n")

    msg.info(f"Max length: {max_length} Min length: {min_length}")
    msg.good(f"Successfully saved {len(files)} documents as .jsonl")


if __name__ == "__main__":
    typer.run(main)
