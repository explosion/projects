"""
Fixes errors in the downloaded data.
"""

import typer

from datasets.dataset import Dataset


def main(dataset_id: str):
    """
    Removes/fixes error in downloaded datasets.
    dataset_id (str): Dataset ID.
    """

    Dataset.generate_dataset_from_id(dataset_id).clean_assets()


if __name__ == '__main__':
    typer.run(main)
