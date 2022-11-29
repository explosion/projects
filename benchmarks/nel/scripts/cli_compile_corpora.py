""" Compiles train/dev/test corpora. """
from pathlib import Path
from typing import Set, Optional

import typer
from datasets.dataset import Dataset


def main(dataset_name: str, language: str, model: str, use_filter_terms: bool = typer.Argument(False)):
    """Create corpora in spaCy format.
    dataset_name (str): Dataset name.
    language (str): Language.
    model (str): Name or path of model with tokenizer, tok2vec and parser.
    use_filter_terms (bool): Whether to use the filter terms defined in the dataset config. If True, only documents
        containing at least one of the specified terms will be included in corpora. If False, all documents are
        included.
    """
    filter_terms: Optional[Set[str]] = None
    if use_filter_terms:
        with open(
            Path(__file__).parent.parent / "wikid" / "configs" / "filter_terms.txt", "r"
        ) as file:
            filter_terms = {ft.replace("\n", "") for ft in file.readlines()}

    # Run name isn't relevant for corpora compilation.
    Dataset.generate_from_id(dataset_name, language).compile_corpora(model, filter_terms)


if __name__ == "__main__":
    typer.run(main)
