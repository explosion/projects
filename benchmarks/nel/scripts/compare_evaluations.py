""" Compare evaluations across multiple runs. """

from datasets.dataset import Dataset
import typer


def main(dataset_name: str):
    """Compare evaluations across all available runs for this dataset.
    dataset_name (str): Name of dataset to evaluate on.
    """

    Dataset.generate_dataset_from_id(dataset_name).compare_evaluations()


if __name__ == "__main__":
    typer.run(main)
