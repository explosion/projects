""" Infer entities for test data. """

from datasets.dataset import Dataset
import typer
from custom_functions import create_candidates_via_embeddings


def main(dataset_name: str):
    """Infer entities for test set.
    dataset_name (str): Name of dataset to evaluate on.
    """
    Dataset.generate_from_id(dataset_name).infer_test_set()


if __name__ == "__main__":
    typer.run(main)
