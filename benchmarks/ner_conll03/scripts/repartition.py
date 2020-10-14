import typer
import random
import shutil
from pathlib import Path


DOC_SEP = "\n\n-DOCSTART- -X- O O\n\n"


def main(in_loc: Path, out_dir: Path, *, dev_part: float=0.2):
    if not out_dir.exists():
        out_dir.mkdir()
    train_docs = (in_loc / "eng.train").open().read().split(DOC_SEP)
    random.shuffle(train_docs)
    cut = int(len(train_docs) * dev_part)
    with (out_dir / "dev.iob").open("w") as file_:
        file_.write(DOC_SEP.join(train_docs[:cut]))

    with (out_dir / "train.iob").open("w") as file_:
        file_.write(DOC_SEP.join(train_docs[cut:]))
    shutil.copyfile(in_loc / "eng.testb", out_dir / "test.iob")


if __name__ == "__main__":
    typer.run(main)
