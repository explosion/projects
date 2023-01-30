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
    ind = np.arange(0, 2)

    def _prepare_data(m_spacy, m_spacy_adj, m_null, m_null_adj):
        def _get_val(enc, dset, ddict):
            avg = round(ddict.get(dset).get(enc).get("f")[0], 2)
            std = round(ddict.get(dset).get(enc).get("f")[1], 2)
            return avg, std

        data = {}
        ddicts = (m_null, m_spacy)
        ddicts_adj = (m_null_adj, m_spacy_adj)
        for dataset in dataset_names:
            data[dataset] = {
                # MultihashEmbed scores
                "mhe_avgs": [
                    _get_val("multihashembed", dataset, ddict)[0] for ddict in ddicts
                ],
                "mhe_stds": [
                    _get_val("multihashembed", dataset, ddict)[1] for ddict in ddicts
                ],
                # Multiembed scores
                "me_avgs": [
                    _get_val("multiembed", dataset, ddict)[0] for ddict in ddicts
                ],
                "me_stds": [
                    _get_val("multiembed", dataset, ddict)[1] for ddict in ddicts
                ],
                # Multihashembed (adjusted scores)
                # "mhe_adj_avgs": [
                #     _get_val("multihashembed", dataset, ddict)[0]
                #     for ddict in ddicts_adj
                # ],
                # "mhe_adj_stds": [
                #     _get_val("multihashembed", dataset, ddict)[1]
                #     for ddict in ddicts_adj
                # ],
            }

        return data

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharey=True)
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
            color="gray",
            edgecolor="k",
            linewidth=1,
            # hatch="/",
        )
        if unseen:
            rects2 = ax.bar(
                ind,
                data.get("mhe_adj_avgs"),
                width,
                yerr=data.get("mhe_adj_stds"),
                label="MultiHashEmbed (adjusted)",
                color="gray",
                edgecolor="k",
                linewidth=1,
                hatch="//",
            )
        rects3 = ax.bar(
            ind + width / offset,
            data.get("me_avgs"),
            width,
            yerr=data.get("me_stds"),
            label="MultiEmbed",
            color="None",
            edgecolor="k",
            # alpha=0.70,
            linewidth=1,
            # hatch="x",
        )

        # Setup ticklabels and legend
        ax.set_ylabel("F1-score", usetex=True)
        if show_xlabel:
            ax.set_xlabel("Dataset", usetex=True)
        if title:
            ax.set_title(title, usetex=True)
        ax.set_xticks(ind)
        # ax.set_xticklabels(dataset_names)
        ax.set_xticklabels(["without", "with"])
        ax.set_ylim(top=1.0)
        if show_legend:
            ax.legend(
                loc="lower center", ncol=3, bbox_to_anchor=legend_loc, frameon=False
            )

        # Hide the right and top splines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        # Add labels for each rectangle
        _autolabel(ax, rects=rects1, xpos="left")
        if unseen:
            _autolabel(ax, rects=rects2, xpos="center")
        _autolabel(ax, rects=rects3, xpos="right")

    # Plot

    data = _prepare_data(
        metrics_spacy,
        metrics_spacy_adjusted,
        metrics_null,
        metrics_null_adjusted,
    )

    for ax, dataset in zip(axs.ravel(), dataset_names):
        _plot(
            ax,
            data.get(dataset),
            title=dataset,
            show_xlabel=False,
            show_legend=False,
            unseen=metrics_spacy_adjusted,
            offset=offset,
        )

    fig.set_size_inches(9, 6)

    # Show single legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc=(0.3, -0.015), frameon=False)

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
    """Plot minimum frequency characterization"""
    msg.info("Plotting MultiEmbed min_freq value")
    metrics_spacy = srsly.read_json(metrics_spacy_path)
    metrics_null = srsly.read_json(metrics_null_path)
    dataset_names = metrics_spacy.keys()

    width = 0.20
    # ind = np.arange(len(dataset_names))
    ind = np.arange(0, 2)

    # def _prepare_data(metrics: Dict) -> Dict[str, Iterable[float]]:
    #     data = {
    #         "10": [],
    #         "5": [],
    #         "1": [],
    #     }
    #     for dataset, scores in metrics.items():
    #         for min_freq in data.keys():
    #             data[min_freq].append(round(scores.get(min_freq), 2))
    #     return data

    def _prepare_data(m_spacy, m_null):
        def _get_val(mf, dset, ddict):
            val = round(ddict.get(dset).get(str(mf)), 2)
            return val

        data = {}
        ddicts = (m_null, m_spacy)
        for dataset in dataset_names:
            data[dataset] = {
                "10": [_get_val(10, dataset, ddict) for ddict in ddicts],
                "5": [_get_val(5, dataset, ddict) for ddict in ddicts],
                "1": [_get_val(1, dataset, ddict) for ddict in ddicts],
            }
        return data

    def _plot(
        ax,
        data: Dict[str, Iterable[float]],
        mhe_data: Iterable[float],
        title: Optional[str] = None,
        show_xlabel: bool = True,
        show_legend: bool = True,
        legend_loc: Tuple[float, float] = (0.5, 0.5),
        offset: int = 1,
    ):
        # breakpoint()
        rects1 = ax.bar(
            ind - width / 0.66667,
            mhe_data,
            width,
            label="MultiHashEmbed",
            linewidth=1,
            color="gray",
            edgecolor="k",
        )
        rects2 = ax.bar(
            ind - width / 2,
            data.get("10"),
            width,
            label="10",
            linewidth=1,
            color="None",
            edgecolor="k",
            # hatch="/",
        )
        rects3 = ax.bar(
            ind + width / 2,
            data.get("5"),
            width,
            label="5",
            linewidth=1,
            color="None",
            edgecolor="k",
            hatch="/",
        )
        rects4 = ax.bar(
            ind + width / 0.66667,
            data.get("1"),
            width,
            label="1",
            # alpha=0.70,
            linewidth=1,
            color="None",
            edgecolor="k",
            hatch="x",
        )

        # Setup ticklabels and legend
        ax.set_ylabel("F1-score", usetex=True)
        if show_xlabel:
            ax.set_xlabel("Dataset", usetex=True)
        if title:
            ax.set_title(title, usetex=True)
        ax.set_xticks(ind)
        ax.set_xticklabels(["without", "with"])
        ax.set_ylim(top=1.0)
        if show_legend:
            ax.legend(
                loc="lower center",
                ncol=4,
                bbox_to_anchor=legend_loc,
                title_fontsize="x-large",
                frameon=False,
            )

        # Hide the right and top splines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        # Add labels for each rectangle
        _autolabel(ax, rects=rects1, xpos="center")
        _autolabel(ax, rects=rects2, xpos="center")
        _autolabel(ax, rects=rects3, xpos="center")
        _autolabel(ax, rects=rects4, xpos="center")

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharey=True)
    if use_tex:
        msg.info("Rendering using LaTeX")
        _use_tex(plt)

    mhe_data = {
        "CoNLL es": [0.74, 0.81],
        "CoNLL nl": [0.69, 0.82],
        "WNUT2017": [0.27, 0.47],
        "Archeology": [0.83, 0.85],
        "AnEM": [0.78, 0.81],
        "OntoNotes": [0.82, 0.86],
    }

    me_data = _prepare_data(metrics_spacy, metrics_null)

    for ax, dataset in zip(axs.ravel(), dataset_names):
        print(dataset)
        _plot(
            ax,
            me_data.get(dataset),
            mhe_data.get(dataset),
            title=dataset,
            show_xlabel=False,
            show_legend=False,
        )

    # Show single legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc=(0.3, -0.010), frameon=False)

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


@app.command("spacy-v-fasttext")
def plot_spacy_v_fasttext(
    # fmt: off
    metrics_path: Path = Arg(..., help="Path to metrics file for spacy vectors"),
    output_path: Optional[Path] = Arg(None, help="Path to save the file (include extension)"),
    use_tex: bool = Opt(False, "--latex", "--tex", "--use-tex", "-t", help="Update plt.rcParams with LaTeX"),
    verbose: bool = Opt(False, "--verbose", "-v", help="Set verbosity."),
    show: bool = Opt(False, "--show", "-S", help="Call plt.show()"),
    title: Optional[str] = Opt(None, help="Set figure title")
    # fmt: on
):
    """Plot comparison between spaCy and fastText"""
    metrics = srsly.read_json(metrics_path)
    dataset_names = metrics.keys()

    width = 0.20
    ind = np.arange(len(dataset_names))

    vectors = ["spacy", "fasttext"]
    data = {
        "spacy_avgs": [],
        "fasttext_avgs": [],
        "spacy_stds": [],
        "fasttext_stds": [],
    }
    for dataset, scores in metrics.items():
        data["spacy_avgs"].append(round(scores.get("spacy").get("f")[0], 2))
        data["spacy_stds"].append(round(scores.get("spacy").get("f")[1], 2))
        data["fasttext_avgs"].append(round(scores.get("fasttext").get("f")[0], 2))
        data["fasttext_stds"].append(round(scores.get("fasttext").get("f")[1], 2))

    fig, ax = plt.subplots(figsize=(10, 3))
    if use_tex:
        msg.info("Rendering using LaTeX")
        _use_tex(plt)

    # Figure configuration
    fig.tight_layout()
    if title:
        fig.suptitle(
            title,
            fontsize="xx-large",
            y=1.05,
            x=0.52,
        )

    rects1 = ax.bar(
        ind - width / 2,
        data.get("spacy_avgs"),
        width,
        yerr=data.get("spacy_stds"),
        label="spaCy (large)",
        linewidth=1,
        color="gray",
        edgecolor="k",
    )
    rects2 = ax.bar(
        ind + width / 2,
        data.get("fasttext_avgs"),
        width,
        yerr=data.get("fasttext_stds"),
        label="fastText",
        linewidth=1,
        color="None",
        edgecolor="k",
    )

    # Setup ticklabels and legend
    ax.set_ylabel("F1-score", usetex=True)
    ax.set_xlabel("Dataset", usetex=True)
    ax.set_xticks(ind)
    ax.set_xticklabels(dataset_names)
    ax.set_ylim(top=1.0)
    ax.legend(
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.5),
        title_fontsize="x-large",
        frameon=False,
    )

    # Hide the right and top splines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    # Add labels for each rectangle
    _autolabel(ax, rects=rects1, xpos="left")
    _autolabel(ax, rects=rects2, xpos="right")

    # Prepare output
    if show:
        plt.show()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        msg.good(f"Saved to {output_path}")


@app.command("num-seeds")
def plot_num_seeds(
    # fmt: off
    metrics_path: Path = Arg(..., help="Path to metrics file for spacy vectors"),
    output_path: Optional[Path] = Arg(None, help="Path to save the file (include extension)"),
    use_tex: bool = Opt(False, "--latex", "--tex", "--use-tex", "-t", help="Update plt.rcParams with LaTeX"),
    verbose: bool = Opt(False, "--verbose", "-v", help="Set verbosity."),
    show: bool = Opt(False, "--show", "-S", help="Call plt.show()"),
    title: Optional[str] = Opt(None, help="Set figure title")
    # fmt: on
):
    """Plot comparison between spaCy and fastText"""
    metrics = srsly.read_json(metrics_path)
    dataset_names = metrics.keys()

    width = 0.20
    ind = np.arange(len(dataset_names))

    vectors = ["spacy", "fasttext"]
    data = {
        "1_avg": [],
        "2_avg": [],
        "3_avg": [],
        "4_avg": [],
        "1_std": [],
        "2_std": [],
        "3_std": [],
        "4_std": [],
    }
    for dataset, scores in metrics.items():
        data["1_avg"].append(round(scores.get("1")[0], 2))
        data["2_avg"].append(round(scores.get("2")[0], 2))
        data["3_avg"].append(round(scores.get("3")[0], 2))
        data["4_avg"].append(round(scores.get("4")[0], 2))

        data["1_std"].append(round(scores.get("1")[1], 2))
        data["2_std"].append(round(scores.get("2")[1], 2))
        data["3_std"].append(round(scores.get("3")[1], 2))
        data["4_std"].append(round(scores.get("4")[1], 2))

    fig, ax = plt.subplots(figsize=(10, 3))
    if use_tex:
        msg.info("Rendering using LaTeX")
        _use_tex(plt)

    # Figure configuration
    fig.tight_layout()
    if title:
        fig.suptitle(
            title,
            fontsize="xx-large",
            y=1.05,
            x=0.52,
        )

    rects1 = ax.bar(
        ind - width / 0.66667,
        data.get("1_avg"),
        width,
        yerr=data.get("1_std"),
        label="1",
        linewidth=1,
        color="None",
        edgecolor="k",
        # hatch="/",
    )
    rects2 = ax.bar(
        ind - width / 2,
        data.get("2_avg"),
        width,
        yerr=data.get("2_std"),
        label="2",
        linewidth=1,
        color="None",
        edgecolor="k",
        hatch="/",
    )
    rects3 = ax.bar(
        ind + width / 2,
        data.get("3_avg"),
        width,
        yerr=data.get("3_std"),
        label="3",
        linewidth=1,
        color="None",
        edgecolor="k",
        hatch="x",
    )
    rects4 = ax.bar(
        ind + width / 0.66667,
        data.get("4_avg"),
        width,
        yerr=data.get("4_std"),
        label="4 (default)",
        alpha=0.70,
        linewidth=1,
        color="gray",
        edgecolor="k",
    )

    # Setup ticklabels and legend
    ax.set_ylabel("F1-score", usetex=True)
    ax.set_xlabel("Dataset", usetex=True)
    ax.set_xticks(ind)
    ax.set_xticklabels(dataset_names)
    ax.set_ylim(top=1.0)
    ax.legend(
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.5),
        title_fontsize="x-large",
        frameon=False,
    )

    # Hide the right and top splines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    # Add labels for each rectangle
    _autolabel(ax, rects=rects1, xpos="center")
    _autolabel(ax, rects=rects2, xpos="center")
    _autolabel(ax, rects=rects3, xpos="center")
    _autolabel(ax, rects=rects4, xpos="center")

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
