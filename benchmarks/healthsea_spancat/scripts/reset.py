import shutil
from pathlib import Path
import typer
from wasabi import Printer

msg = Printer()


def main(path: Path):
    """This script is used to delete directories and reset the project"""
    if path.is_dir():
        answer = input(f"Are you sure you want to reset {path} (y)")
        if answer.lower().strip() == "y":
            try:
                shutil.rmtree(path)
                msg.good(f"Deleted directory {path}")
            except Exception as e:
                print(e)


if __name__ == "__main__":
    typer.run(main)
