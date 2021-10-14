"""Decompress downloaded assets into the specified directory"""

import gzip
import tarfile
import shutil
from pathlib import Path

import typer
from wasabi import msg


def main(src: Path, dest: Path):
    """Decompress files from source to destination

    src (Path): source filepath to decompress
    dest (Path): path to decompress
    """
    if tarfile.is_tarfile(src):
        with tarfile.open(src, "r:gz") as input_file:
            input_file.extractall(dest)
        msg.good(f"Decompressed {src} into {dest}")
    elif src.suffix == ".gz":
        with gzip.open(src, "rb") as input_file:
            with open(dest, "wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
        msg.good(f"Decompressed {src} into {dest}")
    else:
        msg.warn(f"Unknown compression type for file {src}: {src.suffix}")


if __name__ == "__main__":
    typer.run(main)
