"""
Functionality for creating the knowledge base from downloaded assets and by querying Wikipedia's API.
"""

import typer
from datasets.reddit import RedditDataset


def main(dataset_id: str, vectors_model: str):
    """ Create the Knowledge Base in spaCy and write it to file.

     dataset_id (dataset_id): Dataset ID.
     vectors_model (str): Name of model with word vectors to use.
     temp_dir (Path): Path to save knowledge base and NLP pipeline at.
     """

    {"reddit": RedditDataset}[dataset_id]().create_knowledge_base(vectors_model)


if __name__ == "__main__":
    typer.run(main)
