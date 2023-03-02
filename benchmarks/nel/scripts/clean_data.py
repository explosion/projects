"""
Fixes errors in the downloaded data.
"""

import typer

from datasets.dataset import Dataset


def main(dataset_name: str, language: str):
    """
    Removes/fixes error in downloaded datasets.
    dataset_name (str): Dataset name.
    language (str): Language.
    """
    Dataset.generate_from_id(dataset_name, language).clean_assets()


if __name__ == "__main__":
    typer.run(main)
