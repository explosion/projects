""" Compare evaluations across multiple runs. """

from datasets.dataset import Dataset
import typer


def main(dataset_name: str, highlight_criterion: str = "F"):
    """Compare evaluations across all available runs for this dataset.
    dataset_name (str): Name of dataset to evaluate on.
    highlight_criterion (str): Criterion to highlight in table. One of ("F", "r", "p").
    """
    Dataset.generate_from_id(dataset_name).compare_evaluations(highlight_criterion=highlight_criterion)


if __name__ == "__main__":
    typer.run(main)
