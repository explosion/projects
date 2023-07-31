import typer
from pathlib import Path
import glob
import shutil


def main(stem: str, ext: str, input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in glob.glob(str(input_dir.resolve()) + f"/*-{stem}*.{ext}"):
        shutil.copy(filename, str(output_dir.resolve()))


if __name__ == "__main__":
    typer.run(main)
