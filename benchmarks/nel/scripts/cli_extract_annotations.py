""" Parse corpus. """
from datasets.dataset import Dataset
import typer


def main(dataset_name: str, language: str):
    """Parse corpus.
    dataset_name (str): Name of dataset to evaluate on.
    language (str): Language.
    """
    Dataset.generate_from_id(dataset_name, language, "").extract_annotations()


if __name__ == "__main__":
    typer.run(main)
