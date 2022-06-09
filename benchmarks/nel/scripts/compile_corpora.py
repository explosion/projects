""" Compiles train/dev/test corpora. """
import typer

from datasets.dataset import Dataset


def main(dataset_id: str):
    """ Create corpora in spaCy format.
    dataset_id (str): Dataset ID.
    """

    Dataset.generate_dataset_from_id(dataset_id).compile_corpora()


if __name__ == "__main__":
    typer.run(main)
