from pathlib import Path
from typing import Optional, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import srsly
import typer
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option

app = typer.Typer()


GLOBAL_PARAMS = {
    "legend.fontsize": "x-large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "larger",
    "ytick.labelsize": "larger",
}

pylab.rcParams.update(GLOBAL_PARAMS)

COLORS = {
    "spanish_viridian": "#047c5c",
    "wintergreen_dream": "#50957b",
    "morning_blue": "#81ad9b",
    "opal": "#afc5bc",
    "gainsboro": "#dedede",
    "beau_blue": "#bacfdd",
    "dark_sky_blue": "#94c1db",
    "iceberg": "#66b2d9",
    "rich_electric_blue": "#09a4d7",
}


@app.command(name="main-results")
def plot_main_results(
    # fmt: off
    metrics_spacy_path: Path = Arg(..., help="Path to metrics file for spacy vectors"),
    metrics_null_path: Path = Arg(..., help="Path to metrics file for null vectors"),
    metrics_spacy_path_adjusted: Optional[Path] = Opt(None, "--adjusted-spacy", help="Path to metrics file for null vectors (adjusted rows)"),
    metrics_null_path_adjusted: Optional[Path] = Opt(None, "--adjusted-null", help="Path to metrics file for null vectors (adjusted rows)"),
    output_path: Optional[Path] = Arg(None, help="Path to save the file (include extension)"),
    show: bool = Opt(False, "--show", "-S", help="Call plt.show()"),
    use_tex: bool = Opt(False, "--latex", "--tex", "--use-tex", "-t", help="Update plt.rcParams with LaTeX"),
    verbose: bool = Opt(False, "--verbose", "-v", help="Set verbosity."),
    offset: int = Opt(1, "--offset", help="Set offset for the bar chart rects."),
    title: Optional[str] = Opt(None, help="Set figure title")
    # fmt: on
):
    """Plot results between MultiHashEmbed vs. MultiEmbed"""
    msg.info("Plotting MultiHashEmbed vs. MultiEmbed")
    metrics_spacy = srsly.read_json(metrics_spacy_path)
    metrics_null = srsly.read_json(metrics_null_path)
    dataset_names = metrics_spacy.keys()

    metrics_null_adjusted = (
        srsly.read_json(metrics_null_path_adjusted)
        if metrics_null_path_adjusted
        else None
    )
    metrics_spacy_adjusted = (
        srsly.read_json(metrics_spacy_path_adjusted)
        if metrics_spacy_path_adjusted
        else None
    )

    width = 0.20
    ind = np.arange(len(dataset_names))

    def _prepare_data(
        metrics: Dict, adjusted_scores: Optional[Dict] = None
    ) -> Dict[str, Iterable[float]]:
        data = {
            # MultihashEmbed scores
            "mhe_avgs": [],
            "mhe_stds": [],
            # Multiembed scores
            "me_avgs": [],
            "me_stds": [],
            # Multihashembed (adjusted scores)
            "mhe_adj_avgs": [],
            "mhe_adj_stds": [],
        }
        for dataset, scores in metrics.items():
            mhe_avg, mhe_std = scores["multihashembed"].get("f")  # get f-score
            data["mhe_avgs"].append(round(mhe_avg, 2))
            data["mhe_stds"].append(round(mhe_std, 2))
            me_avg, me_std = scores["multiembed"].get("f")  # get f-score
            data["me_avgs"].append(round(me_avg, 2))
            data["me_stds"].append(round(me_std, 2))

            adjusted_report = ""
            if adjusted_scores:
                mhe_adj_avg, mhe_adj_std = adjusted_scores[dataset][
                    "multihashembed"
                ].get("f")
                data["mhe_adj_avgs"].append(round(mhe_adj_avg, 2))
                data["mhe_adj_stds"].append(round(mhe_adj_std, 2))
                adjusted_report = (
                    f"MultiHashEmbed [adjusted] {mhe_adj_avg} ({mhe_adj_std})"
                )

            msg.text(
                f"Dataset `{dataset}`: "
                f"MultiHashEmbed {mhe_avg} ({mhe_std}) "
                f"{adjusted_report} "
                f"MultiEmbed {me_avg} ({me_std})",
                show=verbose,
            )
        return data

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)
    if use_tex:
        msg.info("Rendering using LaTeX")
        _use_tex(plt)

    def _plot(
        ax,
        data: Dict[str, Iterable[float]],
        title: Optional[str] = None,
        show_xlabel: bool = True,
        show_legend: bool = True,
        legend_loc: Tuple[float, float] = (0.5, 0.5),
        unseen: bool = False,
        offset: int = 1,
    ):
        rects1 = ax.bar(
            ind - width / offset,
            data.get("mhe_avgs"),
            width,
            yerr=data.get("mhe_stds"),
            label="MultiHashEmbed",
            color=COLORS.get("spanish_viridian"),
            alpha=0.70,
            linewidth=1,
            edgecolor=COLORS.get("spanish_viridian"),
            hatch="/",
        )
        if unseen:
            rects2 = ax.bar(
                ind,
                data.get("mhe_adj_avgs"),
                width,
                yerr=data.get("mhe_adj_stds"),
                label="MultiHashEmbed (adjusted)",
                color=COLORS.get("opal"),
                alpha=0.70,
                linewidth=1,
                edgecolor=COLORS.get("opal"),
                hatch="//",
            )
        rects3 = ax.bar(
            ind + width / offset,
            data.get("me_avgs"),
            width,
            yerr=data.get("me_stds"),
            label="MultiEmbed",
            color=COLORS.get("rich_electric_blue"),
            alpha=0.70,
            linewidth=1,
            edgecolor=COLORS.get("rich_electric_blue"),
            hatch="x",
        )

        # Setup ticklabels and legend
        ax.set_ylabel("F1-score", usetex=True)
        if show_xlabel:
            ax.set_xlabel("Dataset", usetex=True)
        if title:
            ax.set_title(title, usetex=True)
        ax.set_xticks(ind)
        ax.set_xticklabels(dataset_names)
        ax.set_ylim(top=1.0)
        if show_legend:
            ax.legend(loc="lower center", ncol=3, bbox_to_anchor=legend_loc)

        # Hide the right and top splines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        # Add labels for each rectangle
        _autolabel(ax, rects=rects1, xpos="left")
        if unseen:
            _autolabel(ax, rects=rects2, xpos="center")
        _autolabel(ax, rects=rects3, xpos="right")

    # Plot
    _plot(
        ax1,
        _prepare_data(metrics_spacy, metrics_spacy_adjusted),
        title="with static vectors",
        show_xlabel=False,
        show_legend=False,
        unseen=metrics_spacy_adjusted,
        offset=offset,
    )
    _plot(
        ax2,
        _prepare_data(metrics_null, metrics_null_adjusted),
        title="without static vectors",
        show_legend=True,
        legend_loc=(0.5, -0.7),
        unseen=metrics_null_adjusted,
        offset=offset,
    )

    # Figure configuration
    fig.tight_layout()
    if title:
        fig.suptitle(
            title,
            fontsize="xx-large",
            y=1.05,
            x=0.52,
        )

    # Prepare output
    if show:
        plt.show()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        msg.good(f"Saved to {output_path}")


@app.command("min-freq")
def plot_min_freq(
    # fmt: off
    metrics_spacy_path: Path = Arg(..., help="Path to metrics file for spacy vectors"),
    metrics_null_path: Path = Arg(..., help="Path to metrics file for null vectors"),
    output_path: Optional[Path] = Arg(None, help="Path to save the file (include extension)"),
    use_tex: bool = Opt(False, "--latex", "--tex", "--use-tex", "-t", help="Update plt.rcParams with LaTeX"),
    verbose: bool = Opt(False, "--verbose", "-v", help="Set verbosity."),
    show: bool = Opt(False, "--show", "-S", help="Call plt.show()"),
    title: Optional[str] = Opt(None, help="Set figure title")
    # fmt: on
):
    msg.info("Plotting MultiEmbed min_freq value")
    metrics_spacy = srsly.read_json(metrics_spacy_path)
    metrics_null = srsly.read_json(metrics_null_path)
    dataset_names = metrics_spacy.keys()

    width = 0.20
    ind = np.arange(len(dataset_names))

    def _prepare_data(metrics: Dict) -> Dict[str, Iterable[float]]:
        data = {
            "10": [],
            "5": [],
            "1": [],
        }
        for dataset, scores in metrics.items():
            for min_freq in data.keys():
                data[min_freq].append(round(scores.get(min_freq), 2))
        return data

    def _plot(
        ax,
        data: Dict[str, Iterable[float]],
        title: Optional[str] = None,
        show_xlabel: bool = True,
        show_legend: bool = True,
        legend_loc: Tuple[float, float] = (0.5, 0.5),
        offset: int = 1,
    ):
        rects1 = ax.bar(
            ind - width / offset,
            data.get("10"),
            width,
            label="10",
            color=COLORS.get("spanish_viridian"),
            alpha=0.70,
            linewidth=1,
            edgecolor=COLORS.get("spanish_viridian"),
            hatch="/",
        )
        rects2 = ax.bar(
            ind,
            data.get("5"),
            width,
            label="5",
            color=COLORS.get("opal"),
            alpha=0.70,
            linewidth=1,
            edgecolor=COLORS.get("opal"),
            hatch="//",
        )
        rects3 = ax.bar(
            ind + width / offset,
            data.get("1"),
            width,
            label="1",
            color=COLORS.get("rich_electric_blue"),
            alpha=0.70,
            linewidth=1,
            edgecolor=COLORS.get("rich_electric_blue"),
            hatch="x",
        )

        # Setup ticklabels and legend
        ax.set_ylabel("F1-score", usetex=True)
        if show_xlabel:
            ax.set_xlabel("Dataset", usetex=True)
        if title:
            ax.set_title(title, usetex=True)
        ax.set_xticks(ind)
        ax.set_xticklabels(dataset_names)
        ax.set_ylim(top=1.0)
        if show_legend:
            ax.legend(
                loc="lower center",
                ncol=3,
                bbox_to_anchor=legend_loc,
                title="Minimum frequency",
                title_fontsize="x-large",
            )

        # Hide the right and top splines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        # Add labels for each rectangle
        _autolabel(ax, rects=rects1, xpos="left")
        _autolabel(ax, rects=rects2, xpos="center")
        _autolabel(ax, rects=rects3, xpos="right")

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)
    if use_tex:
        msg.info("Rendering using LaTeX")
        _use_tex(plt)

    # Plot
    _plot(
        ax1,
        _prepare_data(metrics_spacy),
        title="with static vectors",
        show_xlabel=False,
        show_legend=False,
    )
    _plot(
        ax2,
        _prepare_data(metrics_null),
        title="without static vectors",
        show_legend=True,
        legend_loc=(0.5, -0.7),
    )

    # Figure configuration
    fig.tight_layout()
    if title:
        fig.suptitle(
            title,
            fontsize="xx-large",
            y=1.05,
            x=0.52,
        )

    # Prepare output
    if show:
        plt.show()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        msg.good(f"Saved to {output_path}")


def _autolabel(ax, rects, xpos: str = "center"):
    """Attach a text label above each bar in rects, displaying its height.

    xpos indicates which side to place the text w.r.t. the center of
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
            fontsize="large",
        )


def _use_tex(plt):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
        }
    )


if __name__ == "__main__":
    app()
