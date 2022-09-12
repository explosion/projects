""" Evaluation on test data. """

from datasets.dataset import Dataset
import typer
from custom_functions import create_candidates_via_embeddings


def main(dataset_name: str, run_name: str):
    """Evaluate the trained EL component by applying it to unseen text.
    dataset_name (str): Name of dataset to evaluate on.
    run_name (str): Run name.
    """

    Dataset.generate_dataset_from_id(dataset_name).evaluate(
        run_name=run_name, candidate_generation=True, baseline=True, context=True, spacyfishing=False
    )


if __name__ == "__main__":
    typer.run(main)
