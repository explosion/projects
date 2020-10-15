import typer
import csv
import os
from pathlib import Path


def main(root_dir: Path):
    for dataset in root_dir.iterdir():
        print()
        print("dataset", dataset)
        for architecture in dataset.iterdir():
            print()
            print("architecture", architecture)
            results = architecture / "results.tab"
            with results.open("r", encoding="utf8") as csvfile:
                csvreader = csv.reader(csvfile, delimiter="\t")
                header = next(csvreader)
                epoch_index = header.index("epoch")
                step_index = header.index("step")
                metrics = ["cats_micro_f", "cats_macro_f", "cats_macro_auc", ]
                indices = {}
                for metric in metrics:
                    indices[metric] = header.index(metric)
                max_values = {}
                last_values = {}

                for row in csvreader:
                    epoch = row[epoch_index]
                    step = row[step_index]
                    for metric in metrics:
                        index = indices[metric]
                        current_max = max_values.get(metric, 0)
                        current_value = float(row[index])
                        max_values[metric] = current_value if current_value > current_max else current_max
                        last_values[metric] = current_value

                print(f"total steps: {step}")
                print(f"total epochs: {epoch}")
                for metric, value in max_values.items():
                    print(f" BEST {metric}: {round(value, 4)}")
                for metric, value in last_values.items():
                    print(f" LAST {metric}: {round(value, 4)}")


if __name__ == "__main__":
    typer.run(main)