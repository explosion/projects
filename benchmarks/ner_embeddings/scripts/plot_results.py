from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import srsly
import tikzplotlib
import typer
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option

app = typer.Typer()


@app.command(name="main-results")
def plot_main_results(
    # fmt: off
    metrics_path: Path = Arg(..., help="Path to metrics file"),
    output_path: Optional[Path] = Arg(None, help="Path to save the file (include extension)"),
    show: bool = Opt(False, "--show", "-S", help="Call plt.show()"),
    subtitle: str = Opt("with static vectors", "--subtitle", "-s", help="Chart sub-title"),
    verbose: bool = Opt(False, "--verbose", "-v", help="Set verbosity"),
    # fmt: on
):
    """Plot results between MultiHashEmbed vs. MultiEmbed"""
    msg.info("Plotting MultiHashEmbed vs. MultiEmbed")
    metrics = srsly.read_json(metrics_path)

    width = 0.35
    ind = np.arange(len(metrics))

    mhe_avgs, mhe_stds = [], []
    me_avgs, me_stds = [], []
    for dataset, scores in metrics.items():
        mhe_avg, mhe_std = scores["multihashembed"].get("f")  # get f-score
        mhe_avgs.append(round(mhe_avg, 2))
        mhe_stds.append(round(mhe_std, 2))
        me_avg, me_std = scores["multiembed"].get("f")  # get f-score
        me_avgs.append(round(me_avg, 2))
        me_stds.append(round(me_std, 2))
        msg.text(
            f"Dataset `{dataset}`: "
            f"MultiHashEmbed {mhe_avg} ({mhe_std}) "
            f"MultiEmbed {me_avg} ({me_std})",
            show=verbose,
        )

    # fmt: off
    fig, ax = plt.subplots(figsize=(10,8))
    rects1 = ax.bar(ind - width / 2, mhe_avgs, width, yerr=mhe_stds, label="MultiHashEmbed")
    rects2 = ax.bar(ind + width / 2, me_avgs, width, yerr=me_stds, label="MultiEmbed")
    # fmt: on

    ax.set_ylabel("F1-score")
    title = f"MultiHashEmbed vs. MultiEmbed ({subtitle})"
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(metrics.keys())
    ax.set_ylim(top=1.0)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.25))

    _autolabel(ax, rects=rects1, xpos="left")
    _autolabel(ax, rects=rects2, xpos="right")

    fig.tight_layout()
    if show:
        plt.show()
    if output_path:
        plt.savefig(output_path, dpi=300)
        msg.good(f"Saved to {output_path}")


def _autolabel(ax, rects, xpos: str = "center"):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {"center": "center", "right": "left", "left": "right"}
    offset = {"center": 0, "right": 1, "left": -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(offset[xpos] * 3, 3),  # use 3 points offset
            textcoords="offset points",  # in both directions
            ha=ha[xpos],
            va="bottom",
        )


@app.command()
def hi():
    pass


if __name__ == "__main__":
    app()
