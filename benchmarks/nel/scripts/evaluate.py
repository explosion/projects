""" Evaluation on test data. """

from datasets.dataset import Dataset
from custom_functions import create_candidates_via_embeddings


def main(dataset_id: str):
    """ Evaluate the trained EL component by applying it to unseen text. """

    Dataset.generate_dataset_from_id(dataset_id).evaluate(
        candidate_generation=True,
        baseline=False,
        context=False
    )


if __name__ == "__main__":
    # typer.run(main)
    main("reddit")
