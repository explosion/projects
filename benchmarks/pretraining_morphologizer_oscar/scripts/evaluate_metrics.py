from cProfile import label
import typer
from pathlib import Path
import srsly
from wasabi import msg
from os import walk
import matplotlib.pyplot as plt
import numpy as np


def main(metric_folder: Path):

    # Hardcode all datasets
    datasets = {
        "UD_English-EWT": {
            "no_pretraining": {},
            "character_objective": {},
            "vector_objective": {},
            "transformer": {},
        }
    }

    datasets_exist = set()

    # Import all metrics and assign them to a dict
    files = []
    training_epochs = []
    msg.info("Importing all metrics")
    for (dirpath, dirnames, filenames) in walk(metric_folder):
        for filename in filenames:
            for dataset in datasets:
                if str(dataset) in str(filename):
                    datasets_exist.add(dataset)
                    for key in datasets[dataset]:
                        if str(key) in str(filename):
                            if ".jsonl" in str(filename):
                                data = list(srsly.read_jsonl(metric_folder / filename))
                                datasets[dataset][key]["training"] = data
                            else:
                                data = srsly.read_json(metric_folder / filename)
                                datasets[dataset][key]["evaluation"] = data
    msg.good(f"Found metrics for {str(datasets_exist)}")

    # Training eval
    msg.info("Starting evaluation")
    for dataset in datasets:
        x_list = []
        y_list = []
        name_list = []
        for metric_type in datasets[dataset]:
            epochs = []
            scores = []
            for line in datasets[dataset][metric_type]["training"]:
                epochs.append(line["epoch"])
                scores.append(line["score"])
            x_list.append(epochs)
            y_list.append(scores)
            name_list.append(metric_type)

        for x, y, name in zip(x_list, y_list, name_list):
            plt.plot(x, y, label=name)

            for _x, _y in zip(x, y):
                label = "{:.2f}".format(_y)
                plt.annotate(label, (_x, _y + 0.005), size=7, ha="center")

        # Plot settings
        ax = plt.gca()
        ax.set_ylim([0.8, 1.0])
        ax.set_ylabel("Score")
        ax.set_xlabel("Epochs")
        plt.legend()
        plt.grid()
        plt.title(f"Training {dataset}", size=15)
        plt.savefig(metric_folder / f"{dataset}_training_graph.png", dpi=300)
        msg.good(f"Saved training plot for {dataset}")


if __name__ == "__main__":
    typer.run(main)
