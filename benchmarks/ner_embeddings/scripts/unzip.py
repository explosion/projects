import shutil
import gzip
import typer
from pathlib import Path
from typing import Optional
from spacy.util import ensure_path


def unzip(input_file: Path, output_file: Optional[Path] = None):
    input_file = ensure_path(input_file)
    if not input_file.exists():
        raise ValueError(f"Could not find {input_file}.")
    if output_file is None:
        if not input_file.suffix:
            msg.fail("Please provide output_file")
        else:
            output_file = input_file.parents[0] / input_file.stem
    else:
        output_file = ensure_path(output_file)
    if output_file.exists():
        raise ValueError(f"Output file already exists {output_file}.")
    with gzip.open(input_file, "rb") as fin:
        with open(output_file, "wb") as fout:
            shutil.copyfileobj(fin, fout)


if __name__ == "__main__":
    typer.run(unzip)

