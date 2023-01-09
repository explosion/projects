import gzip
import shutil
import tarfile
from pathlib import Path
from typing import Optional

import typer
from spacy.util import ensure_path
from wasabi import Printer

app = typer.Typer()
msg = Printer()


@app.command()
def untar(input_file: Path, output_dir: Optional[Path] = None):
    """
    Untar all contents of input_file into
    output_dir. If output_dir is not provided
    extract all contents into the input_file's
    parent directory.
    """
    input_path = ensure_path(input_file)
    if output_dir is None:
        output_path = input_path.parents[0]
    else:
        output_path = ensure_path(output_dir)
    if not input_path.exists():
        raise ValueError(f"Could not find {input_file}.")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    compressed = tarfile.open(input_path)
    compressed.extractall(output_path)
    compressed.close()


@app.command()
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


@app.command()
def reencode(
    input_path: Path, output_path: Path, source_encoding: str, target_encoding: str
) -> None:
    input_path = ensure_path(input_path)
    output_path = ensure_path(output_path)
    if not input_path.exists():
        raise ValueError(f"Could not find {input_path}.")
    if output_path.exists():
        raise ValueError(f"Output file already exists {output_path}.")
    with open(input_path, "r", encoding=source_encoding) as fin:
        content = fin.read()
    with open(output_path, "w", encoding=target_encoding) as fout:
        fout.write(content)


if __name__ == "__main__":
    app()
