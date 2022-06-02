"""
Fixes errors in the downloaded data.
"""

import fileinput
import os.path
from pathlib import Path

import typer


def main():
    """
    Removes/fixes error in downloaded datasets.
    """

    asset_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "assets"

    # Reddit EL dataset

    to_remove = {
        "bronze_post_annotations.tsv": {},
        "bronze_comment_annotations.tsv": {}
    }
    to_replace = {
        "bronze_comment_annotations.tsv": {
            "`` How to Lose a Guy in 10 Days": "How to Lose a Guy in 10 Days",
            "in 10 Days\t": "in 10 Days\"\t",
            "Money\t325\t330\tmoney\n": "Money\t323\t328\tmoney\n"
        },
        "bronze_post_annotations.tsv": {
            # "8qntul\tsports\tad\tAdvertising\t56\t58\tparade\n": "8qntul\tsports\tad\tParade\t56\t58\tparade\n",
            # "dtjq8bn\tpolitics\tGOOD\tGood\t102\t106\tGOOD\n": "dtjq8bn\tpolitics\tGOOD\tGood\t94\t98\tGOOD\n"
        }
    }

    for file_name in to_replace:
        for line in fileinput.input(asset_root / "reddit" / file_name, inplace=True):
            if line not in to_remove[file_name]:
                new_line = line
                for snippet in to_replace[file_name]:
                    if snippet in line:
                        new_line = line.replace(snippet, to_replace[file_name][snippet])
                        break
                print(new_line, end='')


if __name__ == '__main__':
    typer.run(main)
