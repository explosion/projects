""" Evaluation on test data. """
from datasets.dataset import Dataset
import typer
from custom_functions import create_candidates_via_embeddings


def main(dataset_name: str, run_name: str, language: str):
    """Evaluate the trained EL component by applying it to unseen text.
    dataset_name (str): Name of dataset to evaluate on.
    run_name (str): Run name.
    language (str): Language.
    """
    Dataset.generate_from_id(dataset_name, language, run_name).evaluate(run_name=run_name)


if __name__ == "__main__":
    main("mewsli_9", "cg-lexical", "en")
    # typer.run(main)
