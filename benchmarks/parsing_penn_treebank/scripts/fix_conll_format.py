import typer
from pathlib import Path


def main(in_loc: Path, out_loc: Path):
    """Fix the POS tag column in the test file. In this format it provides
    the predicted tags, it should provide coarse tags.
    """
    fixed = []
    with in_loc.open() as input_file:
        with out_loc.open("w") as output_file:
            for line in input_file:
                if not line.strip():
                    output_file.write(line)
                else:
                    pieces = line.split()
                    pieces[3] = "X"
                    output_file.write("\t".join(pieces) + "\n")


if __name__ == "__main__":
    typer.run(main)
