"""Retrieve candidates for mentions in corpus."""
import typer as typer

from datasets.dataset import Dataset


def main(dataset_name: str, language: str):
    """Retrieve candidates for mentions in corpus.
    dataset_name (str): Name of dataset to evaluate on.
    language (str): Language.
    """
    Dataset.generate_from_id(dataset_name, language, "").retrieve_candidates_for_mentions()


if __name__ == "__main__":
    typer.run(main)
