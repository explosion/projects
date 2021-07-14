import argparse
import shutil
from os import PathLike
from pathlib import Path
from typing import Union


def mkdir(path: Union[PathLike, str]):
    Path(path).mkdir(parents=True, exist_ok=True)


def mv(src: Union[PathLike, str], dst: Union[PathLike, str]):
    shutil.move(src, dst)


def _rm(item: Path):
    if item.is_file():
        item.unlink()
    elif item.is_dir():
        shutil.rmtree(item)


def rm(path: Union[PathLike, str]):
    item = Path(path)

    if path.endswith(r"*"):
        parent = Path(path[:-1])
        if parent.is_dir():
            for desc in parent.rglob("*"):
                _rm(desc)
        else:
            _rm(item)
    else:
        _rm(item)


if __name__ == '__main__':
    cparser = argparse.ArgumentParser(description="Utility functions for file manipulation in spaCy projects")
    csubparsers = cparser.add_subparsers()

    parser_mv = csubparsers.add_parser("mv",
                                       help="Move or rename file. Destination directory will be created if"
                                            " it does not exist yet")
    parser_mv.set_defaults(func=mv)
    parser_mv.add_argument("src", help="File to move")
    parser_mv.add_argument("dst", help="Destination to move to")

    parser_rm = csubparsers.add_parser("rm",
                                       help="Remove file or directory. If the given path ends with * all files in its"
                                            " parent directory will be removed.")
    parser_rm.set_defaults(func=rm)
    parser_rm.add_argument("path", help="Directory or file to remove")

    parser_mkdir = csubparsers.add_parser("mkdir",  help="Creates a given directory (and its parents if needed).")
    parser_mkdir.set_defaults(func=mkdir)
    parser_mkdir.add_argument("path", help="Directory to create")

    cargs = vars(cparser.parse_args())
    func = cargs.pop("func")
    func(**cargs)

