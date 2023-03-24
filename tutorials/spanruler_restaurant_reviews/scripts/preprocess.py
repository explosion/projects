from pathlib import Path

import typer

Arg = typer.Argument


def preprocess(
    input_path: Path = Arg(..., help="Input path for the raw IOB files."),
    output_path: Path = Arg(..., help="Output path for the processed IOB files."),
):
    """Preprocess the raw IOB files from MIT Restaurant Reviews

    The IOB format from the MIT Restaurant reviews dataset has the tokens and
    its annotations flipped, without any delimiters. This command cleans
    them up so that they can easily be passed to `spacy convert`
    """
    with input_path.open("r", encoding="utf-8") as infile:
        input_lines = infile.read().splitlines()

    # Here, the annotations and tokens are separated by '\t'
    # we want them in reverse.
    annotation_token = [line.split("\t") for line in input_lines]

    with output_path.open("w", encoding="utf-8") as outfile:
        for lines in annotation_token:
            if len(lines) == 2:  # contains both annotation and token
                annotation, token = lines
                outfile.write(f"{token}\t{annotation}\n")
            if len(lines) == 1:  # contains space
                outfile.write("\n")


if __name__ == "__main__":
    typer.run(preprocess)
