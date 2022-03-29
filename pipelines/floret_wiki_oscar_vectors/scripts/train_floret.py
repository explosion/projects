import typer
from pathlib import Path
import floret


def main(
    input_file: Path,
    output_stem: str,
    mode: str = "floret",
    model: str = "cbow",
    dim: int = 300,
    mincount: int = 10,
    minn: int = 5,
    maxn: int = 5,
    neg: int = 10,
    epoch: int = 5,
    hashcount: int = 2,
    bucket: int = 50000,
    thread: int = 8,
    lr: float = 0.05,
):
    floret_model = floret.train_unsupervised(
        str(input_file.absolute()),
        model=model,
        mode=mode,
        dim=dim,
        minCount=mincount,
        minn=minn,
        maxn=maxn,
        neg=neg,
        epoch=epoch,
        hashCount=hashcount,
        bucket=bucket,
        thread=thread,
        lr=lr,
    )
    floret_model.save_model(output_stem + ".bin")
    floret_model.save_vectors(output_stem + ".vec")
    if mode == "floret":
        floret_model.save_floret_vectors(output_stem + ".floret")


if __name__ == "__main__":
    typer.run(main)
