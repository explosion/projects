"""
Functionality for creating the knowledge base from downloaded assets and by querying Wikipedia's API.
"""

import typer
from datasets.dataset import Dataset


def main(dataset_name: str, vectors_model: str, n_kb_tokens: int):
    """ Create the Knowledge Base in spaCy and write it to file.
    dataset_name (str): Dataset name.
    vectors_model (str): Name of model with word vectors to use.
    n_kb_tokens (int): Number of tokens in entity description to include when inferring embedding.
    """

    Dataset.generate_dataset_from_id(dataset_name).create_knowledge_base(vectors_model, n_kb_tokens)


if __name__ == "__main__":
    typer.run(main)
