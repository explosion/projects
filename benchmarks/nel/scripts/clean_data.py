"""
Fixes errors in the downloaded data.
"""

import fileinput
import os.path
import typer


def main(dataset_id_reddit: str):
    """
    Removes/fixes error in downloaded datasets.
    dataset_id_reddit (str): ID for Reddit dataset.
    """

    # Reddit EL dataset

    to_remove = {
    }
    to_replace = {
        "bronze_post_annotations.tsv": {
            "dyt7ok4\tmovies\t\"How to Lose a Guy in 10 Days\tHow_to_Lose_a_Guy_in_10_Days\t73\t102\t`` How to Lose a "
            "Guy in 10 Days\n": "dyt7ok4\tmovies\tHow to Lose a Guy in 10 Days\tHow_to_Lose_a_Guy_in_10_Days\t74\t102"
                                "\tHow to Lose a Guy in 10 Days\n",
            "8qntul\tsports\tad\tAdvertising\t56\t58\tparade\n": "8qntul\tsports\tad\tParade\t56\t58\tparade\n",
            "dtjq8bn\tpolitics\tGOOD\tGood\t102\t106\tGOOD\n": "dtjq8bn\tpolitics\tGOOD\tGood\t94\t98\tGOOD\n"
        },
        "silver_post_annotations.tsv": {
            "duwo7ml\tpolitics\tmoney\tMoney\t325\t330\tmoney\n": "duwo7ml\tpolitics\tmoney\tMoney\t323\t328\tmoney\n"
        }
    }

    for line in fileinput.input(os.path.join("assets", dataset_id_reddit, "bronze_post_annotations.tsv"), inplace=True):
        if line not in to_remove:
            # Writes back line into the original file.
            print(to_replace.get(line, line), end='')


if __name__ == '__main__':
    typer.run(main)
