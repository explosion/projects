""" Compiles train/dev/test corpora. """
import typer
from datasets.dataset import Dataset


def main(dataset_name: str):
    """Create corpora in spaCy format.
    dataset_name (str): Dataset name.
    """

    Dataset.generate_dataset_from_id(dataset_name).compile_corpora()


if __name__ == "__main__":
    typer.run(main)
