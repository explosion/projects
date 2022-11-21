""" Compiles train/dev/test corpora. """

import typer
from datasets.dataset import Dataset
from wikid import read_filter_terms


def main(dataset_name: str, language: str, use_filter_terms: bool = typer.Argument(False)):
    """Create corpora in spaCy format.
    dataset_name (str): Dataset name.
    language (str): Language.
    use_filter_terms (bool): Whether to use the filter terms defined in the dataset config. If True, only documents
        containing at least one of the specified terms will be included in corpora. If False, all documents are
        included.
    """
    # Run name isn't relevant for corpora compilation.
    Dataset.generate_from_id(dataset_name, language).compile_corpora(read_filter_terms() if use_filter_terms else None)


if __name__ == "__main__":
    typer.run(main)
