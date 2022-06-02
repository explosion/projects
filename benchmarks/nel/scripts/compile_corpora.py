""" Compiles train/dev/test corpora. """
import typer
from datasets.reddit import RedditDataset


def main(dataset_id: str):
    """ Create corpora in spaCy format.
    dataset_id (str): Dataset ID.
    """

    {"reddit": RedditDataset}[dataset_id]().create_corpora()


if __name__ == "__main__":
    typer.run(main)
    # main(
    #     "reddit",
    #     Path("/home/raphael/dev/spacy-projects/benchmarks/nel/temp")
    # )
