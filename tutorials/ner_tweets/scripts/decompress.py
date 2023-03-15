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
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(input_file, dest)
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
