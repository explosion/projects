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
        },
        "UD_German-HDT": {
            "no_pretraining": {},
            "character_objective": {},
            "vector_objective": {},
            "transformer": {},
        },
        "UD_Dutch-Alpino": {
            "no_pretraining": {},
            "character_objective": {},
            "vector_objective": {},
            "transformer": {},
        },
    }

    datasets_exist = set()

    # Import all metrics and assign them to a dict
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

    del_list = []
    for dataset in datasets:
        if dataset not in datasets_exist:
            del_list.append(dataset)
    for del_key in del_list:
        del datasets[del_key]

    del_list_experiment = []
    for dataset in datasets:
        for experiment in dataset:
            if len(experiment) == 0:
                del_list_experiment.append((dataset, experiment))
    for del_experiment in del_list_experiment:
        del datasets[del_experiment[0]][del_experiment[1]]

    # Training eval
    msg.info("Starting training evaluation")
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

    # Evaluation comparison
    msg.info("Starting evaluation comparison")

    # Set metrics which we want to compare
    compare_metrics = [
        "pos_acc",
        "morph_micro_p",
        "morph_micro_f",
        "morph_micro_r",
        "morph_per_feat",
        "speed",
    ]

    metric_types = [
        "no_pretraining",
        "character_objective",
        "vector_objective",
        "transformer",
    ]
    for dataset in datasets:
        metric_table = {}
        for metric in compare_metrics:
            metric_table[metric] = {}
            for metric_type in datasets[dataset]:
                eval_data = datasets[dataset][metric_type]["evaluation"]

                if type(eval_data[metric]) == dict:
                    for label in eval_data[metric]:
                        if label not in metric_table:
                            metric_table[label] = {}
                        metric_table[label][metric_type] = eval_data[metric][label]["f"]
                else:
                    metric_table[metric][metric_type] = eval_data[metric]

        dataset_output = ""
        header = "| Label |"
        row_sep = "| :----: |"
        for metric in metric_types:
            header += f" {metric} |"
            row_sep += " :----: |"
        dataset_output += header + "\n"
        dataset_output += row_sep + "\n"

        for metric in metric_table:
            row = f"| {metric} |"
            if len(metric_table[metric]) > 0:
                no_pretraining_value = 0
                for metric_type in metric_types:
                    difference = 0
                    if metric_type == "no_pretraining":
                        no_pretraining_value = metric_table[metric][metric_type]
                    else:
                        difference = (
                            metric_table[metric][metric_type] - no_pretraining_value
                        )

                    value = "{:.2f}".format(metric_table[metric][metric_type])
                    difference = "{:.2f}".format(difference)
                    row += f" {value} ({difference}) |"

                dataset_output += row + "\n"

        with open(
            metric_folder / f"{dataset}_evaluation_comparison.md", "w", encoding="utf-8"
        ) as f:
            f.write(dataset_output)

        msg.good(f"Saved eval comparison for {dataset}")
    msg.info(f"Done!")


if __name__ == "__main__":
    typer.run(main)
