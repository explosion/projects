from pathlib import Path
from typing import List, Optional

import pandas as pd
import srsly
import typer
from wasabi import msg

from .constants import CONFIGS, DATASET_VECTORS


def _create_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """Create dataframe with average and stdev"""
    avg = input_df.mean(axis=0)
    std = input_df.std(axis=0, ddof=0)
    df = avg.to_frame("avg")
    df["std"] = std.to_frame("std")
    return df


def main(
    # fmt: off
    input_dir: Path = typer.Argument(..., help="Path to the parent metrics directory", dir_okay=True, exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "--output-dir", "-o", help="Output directory to save the collated results."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print the results during the run.")
    # fmt: on
):
    """Collate results from multiple trials"""

    dataset_dirs: List[Path] = []
    for dataset, meta in DATASET_VECTORS.items():
        for config in CONFIGS:
            for vectors in ("null", meta.get("spacy")):
                dataset_dir = Path(input_dir) / dataset / vectors / config
                if dataset_dir.is_dir():
                    dataset_dirs.append(dataset_dir)

    msg.info(f"Found {len(dataset_dirs)} experiment/s in {input_dir}")
    msg.text([str(d) for d in dataset_dirs], show=verbose)

    for dataset_dir in dataset_dirs:
        trial_dirs = list(dataset_dir.iterdir())
        msg.info(f"Found {len(trial_dirs)} trials for {dataset_dir}", show=verbose)
        scores = [srsly.read_json(list(t.glob("*.json"))[0]) for t in trial_dirs]
        df = pd.DataFrame.from_dict(scores).dropna(axis=1)

        # Compute top-level metrics
        relevant_cols = ["ents_p", "ents_r", "ents_f", "speed"]
        top_level = df[relevant_cols]
        top_level_df = _create_df(top_level)

        if verbose:
            # We're using print instead of msg because print()
            # renders a table-like appearance.
            print(top_level_df)

        # Save results
        if output_dir:
            data = top_level_df.to_dict()
            _output_dir = output_dir / dataset_dir.relative_to(input_dir)
            _output_dir.mkdir(parents=True, exist_ok=True)
            output_path = _output_dir / "metrics_summary.json"
            srsly.write_json(output_path, data)
            msg.info(f"Saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
