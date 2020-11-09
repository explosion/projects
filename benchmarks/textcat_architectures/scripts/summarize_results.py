import typer
import csv
from pathlib import Path


def main(root_dir: Path):
    for dataset in root_dir.iterdir():
        print()
        print(f"DATASET: {dataset.name}")
        for architecture in dataset.iterdir():
            print()
            print(f" architecture: {architecture.name}")
            results = architecture / "results.tab"
            with results.open("r", encoding="utf8") as csvfile:
                csvreader = csv.reader(csvfile, delimiter="\t")
                header = next(csvreader)
                epoch_index = header.index("epoch")
                step_index = header.index("step")
                metrics = ["cats_micro_f", "cats_macro_f", "cats_macro_auc"]
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
                        max_values[metric] = (
                            current_value
                            if current_value > current_max
                            else current_max
                        )
                        last_values[metric] = current_value

                print(f"  total steps: {step}")
                print(f"  total epochs: {epoch}")
                print(f"  best results:")
                for metric, value in max_values.items():
                    print(
                        f"   {metric}: {round(value, 4)}   (LAST: {round(last_values[metric], 4)})"
                    )
            print()


if __name__ == "__main__":
    typer.run(main)
