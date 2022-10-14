import typer
from pathlib import Path
from typing import Optional
from wasabi import msg
import srsly
import pandas as pd


def _create_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """Create dataframe with average and stdev"""
    avg = input_df.mean(axis=0)
    std = input_df.std(axis=0)
    df = avg.to_frame("avg")
    df["std"] = std.to_frame("std")
    return df


def main(
    # fmt: off
    input_dir: Path = typer.Argument(..., help="Path to the metrics directory", dir_okay=True, exists=True),
    output_path: Optional[Path] = typer.Option(None, "--output", "--output-path", "-o", help="Filepath to save the collated results."),
    # fmt: on
):
    """Collate results from multiple trials"""
    trial_dirs = list(input_dir.iterdir())
    msg.info(f"Found {len(trial_dirs)} trials for {input_dir}")
    scores = [srsly.read_json(list(t.glob("*.json"))[0]) for t in trial_dirs]
    df = pd.DataFrame.from_dict(scores).dropna(axis=1)

    # Compute top-level metrics
    # Here, I select the columns that have float values
    top_level = df.select_dtypes(include=["float64"])
    top_level_df = _create_df(top_level)

    print(top_level_df)
    if output_path:
        data = top_level_df.to_dict()
        srsly.write_json(output_path, data)
        msg.info(f"Saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
