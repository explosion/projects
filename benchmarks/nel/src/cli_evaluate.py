""" Evaluation on test data. """
from typing import Optional

from datasets.dataset import Dataset
import custom_functions
import typer


def main(dataset_name: str, run_name: str, language: str, gpu_id: Optional[int] = typer.Argument(None)):
    """Evaluate the trained EL component by applying it to unseen text.
    dataset_name (str): Name of dataset to evaluate on.
    run_name (str): Run name.
    language (str): Language.
    gpu_id (Optional[int]): ID of GPU to utilize for evaluation.
    """
    Dataset.generate_from_id(dataset_name, language, run_name).evaluate(gpu_id=gpu_id)


if __name__ == "__main__":
    # main("mewsli_9", "default", "en", 0)
    typer.run(main)
