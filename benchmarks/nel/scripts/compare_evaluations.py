""" Compare evaluations across multiple runs. """

from datasets.dataset import Dataset
import typer


def main(dataset_name: str, highlight_criterion: str = "F-score"):
    """Compare evaluations across all available runs for this dataset.
    dataset_name (str): Name of dataset to evaluate on.
    highlight_criterion (str): Criterion to highlight in table. One of ("F-score", "Recall", "Precision").
    """

    Dataset.generate_dataset_from_id(dataset_name).compare_evaluations(highlight_criterion=highlight_criterion)


if __name__ == "__main__":
    main("mewsli_9", highlight_criterion="Precision")
    # typer.run(main)
