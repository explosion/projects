""" Evaluation on test data. """
from datasets.dataset import Dataset
import typer


def main(dataset_name: str, run_name: str, language: str):
    """Evaluate the trained EL component by applying it to unseen text.
    dataset_name (str): Name of dataset to evaluate on.
    run_name (str): Run name.
    language (str): Language.
    """
    # todo
    #   - add custom loader making sure that, for training, entities are in documents loaded from docbin
    #   - figure out spacy.load() issue (issue with to_disk() in combination with wikikb?)
    Dataset.generate_from_id(dataset_name, language, run_name).evaluate(run_name=run_name)


if __name__ == "__main__":
    main("mewsli_9", "cg-default", "en")
    # typer.run(main)
