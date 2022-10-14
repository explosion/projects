"""
Fixes errors in the downloaded data.
"""

import typer

from datasets.dataset import Dataset


def main(dataset_name: str):
    """
    Removes/fixes error in downloaded datasets.
    dataset_name (str): Dataset name.
    """
    Dataset.generate_from_id(dataset_name).clean_assets()


if __name__ == "__main__":
    typer.run(main)
