from pathlib import Path
from string import digits

import typer
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def preprocess(
        input_path: Path,
        output_path: Path,
        *,
        strip_digits: bool = False
) -> None:
    """
    Helper function to canonicalize all datasets into the same ConLL format.
    """
    with input_path.open() as f:
        lines = f.readlines()
    with output_path.open("w") as f:
        for i, line in enumerate(lines):
            new_line = line
            if strip_digits:
                new_line = line if line == "\n" else line.lstrip(digits)[1:]
            if new_line in {"-DOCSTART-\tO\n", "-DOCSTART- -DOCSTART- O\n"}:
                new_line = "-DOCSTART- -X- O O\n"
            elif new_line == "\t\n":
                new_line = "\n"
            f.write(new_line)
    msg.good(f"Saved preprocessed data to {output_path}")


if __name__ == "__main__":
    typer.run(preprocess)
