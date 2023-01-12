"""Compare the three metrics of the trained model to determine the boost/degradation in performance"""
import srsly
import typer
from pathlib import Path
from wasabi import msg
import json

import spacy
from spacy.tokens import DocBin
from spacy import Language

METRICS_TO_TRACK = [
    "cats_score",
]


def compare_metrics(
    texcat_metric_path: Path, update_metric_path: Path, rehearse_metric_path: Path
):
    textcat_metrics = srsly.read_json(texcat_metric_path)
    update_metrics = srsly.read_json(update_metric_path)
    rehearse_metrics = srsly.read_json(rehearse_metric_path)

    msg.info("Start comparing metric files.")
    msg.good("All files loaded")
    header = (
        "Metric",
        "Base",
        "Update",
        "Rehearsal",
    )
    aligns = ("l", "r", "r", "r")
    data = []

    update_improvement = 0
    rehearse_improvement = 0

    for label_metric in textcat_metrics["cats_f_per_type"]:
        textcat_f = round(textcat_metrics["cats_f_per_type"][label_metric]["f"], 3)
        update_f = round(update_metrics["cats_f_per_type"][label_metric]["f"], 3)
        rehearse_f = round(rehearse_metrics["cats_f_per_type"][label_metric]["f"], 3)

        update_difference = calculate_percentage(textcat_f, update_f)
        update_improvement += update_difference
        rehearse_difference = calculate_percentage(textcat_f, rehearse_f)
        rehearse_improvement += rehearse_difference

        data.append(
            (
                label_metric,
                f"{textcat_f}",
                f"{update_f} ({update_difference})",
                f"{rehearse_f} ({rehearse_difference})",
            )
        )

    data.append(
        (
            "Total improvement",
            f"",
            f"{round(update_improvement,3)}",
            f"{round(rehearse_improvement,3)}",
        )
    )

    msg.table(data, header=header, divider=True, aligns=aligns)


def calculate_percentage(metric_A: float, metric_B: float):
    difference = metric_B - metric_A
    # percentage = ((metric_B / metric_A) * 100) - 100
    return round(difference, 3)


if __name__ == "__main__":
    typer.run(compare_metrics)
