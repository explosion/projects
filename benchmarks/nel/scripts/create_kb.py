"""
Functionality for creating the knowledge base from downloaded assets and by querying Wikipedia's API.
"""

import typer
from datasets.dataset import Dataset


def main(dataset_id: str, vectors_model: str):
    """ Create the Knowledge Base in spaCy and write it to file.

     dataset_id (dataset_id): Dataset ID.
     vectors_model (str): Name of model with word vectors to use.
     temp_dir (Path): Path to save knowledge base and NLP pipeline at.
     """

    Dataset.generate_dataset_from_id(dataset_id).create_knowledge_base(vectors_model)


if __name__ == "__main__":
    # typer.run(main)
    main("reddit", "en_core_web_md")
