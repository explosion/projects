from typing import Optional, List
import typer
import shutil
from pathlib import Path


def main(
    output_file: Path,
    input_file: Optional[List[Path]] = typer.Option(None),
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as output_fileh:
        for filename in input_file:
            with open(filename, "rb") as fileh:
                shutil.copyfileobj(fileh, output_fileh, length=1000000)


if __name__ == "__main__":
    typer.run(main)
