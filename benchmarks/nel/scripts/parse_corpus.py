""" Parse corpus. """
from datasets.dataset import Dataset
import typer


def main(dataset_name: str, run_name: str, language: str):
    """Parse corpus.
    dataset_name (str): Name of dataset to evaluate on.
    run_name (str): Run name.
    language (str): Language.
    """
    Dataset.generate_from_id(dataset_name, language, run_name).parse_corpus(run_name=run_name)


if __name__ == "__main__":
    typer.run(main)
