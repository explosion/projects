import gzip
import shutil
from pathlib import Path
from typing import Optional

import typer
from spacy.util import ensure_path
from wasabi import msg

app = typer.Typer()


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
def copy_contents(input_dir: Path, output_dir: Path):
    input_dir = ensure_path(input_dir)
    output_dir = ensure_path(output_dir)
    if not input_dir.exists():
        raise ValueError(f"Could not find {input_dir}.")
    elif not output_dir.exists():
        raise ValueError(f"Could not find {output_dir}.")
    elif not input_dir.is_dir():
        raise ValueError("'input_dir' must be a directory")
    elif not output_dir.is_dir():
        raise ValueError("'output_dir' must be a directory")
    for path in input_dir.iterdir():
        if path.is_dir():
            shutil.copytree(path, output_dir)
        else:
            shutil.copy(path, output_dir)


@app.command()
def remove(input_path: Path):
    input_path = ensure_path(input_path)
    if not input_path.exists():
        raise ValueError(f"Could not find {input_path}.")
    if input_path.is_dir():
        try:
            shutil.rmtree(input_path)
        except PermissionError:
            msg.warn(
                f"Do not seem to have permission to remove {input_path}. "
                "This is a known issue that can occur on Windows."
            )
    else:
        try:
            input_path.unlink()
        except PermissionError:
            msg.warn(
                f"Do not seem to have permission to remove {input_path}. "
                "This is a known issue that can occur on Windows."
            )


@app.command()
def expected_arg(config: str, expected: str, given: str):
    if given != expected:
        msg.warn(f"For the argument '{config}' "
                 f"expected '{expected}', but found '{given}'.", exits=1)


if __name__ == "__main__":
    app()
