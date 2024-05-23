from __future__ import annotations

import json
import math
import pickle
import time
import traceback
from argparse import ArgumentParser, Namespace
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import wait as wait_futures
from pathlib import Path

import numpy as np
from typing_extensions import Literal

from .plot_styles import (
    ALGORITHMS,
    BENCH_TABLE_NAMES,
    BENCHMARK_COLORS,
    CUSTOM_MARKERS,
    COLOR_MARKER_DICT,
    DATASETS,
    RC_PARAMS,
    X_LABEL,
    Y_LABEL,
    Y_LIMITS,
    get_xticks,
)
from .plotting_types import ExperimentResults, all_possibilities, fetch_results

HERE = Path(__file__).parent.absolute()
DEFAULT_BASE_PATH = HERE.parent.parent
DEFAULT_RESULTS_PATH = HERE.parent / "results"

is_last_row = lambda idx, nrows, ncols: idx >= (nrows - 1) * ncols
HERE = Path(__file__).parent.absolute()
is_first_column = lambda idx, ncols: idx % ncols == 0

class with_traceback:

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            raise type(e)(f"{tb}\n{str(e)}") from e


def now() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def reorganize_legend(
    fig,
    axs,
    to_front: list[str],
    bbox_to_anchor: tuple[float, float],
    ncol: int,
    fontsize: int = 22,
) -> None:
    import matplotlib.pyplot as plt

    ax = axs if isinstance(axs, plt.Axes) else axs[0]
    handles, labels = ax.get_legend_handles_labels()
    handles_to_plot, labels_to_plot = [], []  # type: ignore
    handles_default, labels_default = [], []  # type: ignore
    for h, l in zip(handles, labels):
        if l not in (labels_to_plot + labels_default):
            if l.lower() in to_front:
                handles_default.append(h)
                labels_default.append(l)
            else:
                handles_to_plot.append(h)
                labels_to_plot.append(l)

    handles_to_plot = handles_default + handles_to_plot
    labels_to_plot = labels_default + labels_to_plot

    leg = fig.legend(
        handles_to_plot,
        labels_to_plot,
        fontsize=fontsize,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=True,
        markerscale=2,
    )

    for legend_item in leg.legendHandles:
        legend_item.set_linewidth(2.0)


def plot_relative_ranks(
    algorithms: list[str],
    filepath: Path,
    yaxis: str,
    xaxis: str,
    subtitle_results: dict[str, ExperimentResults],
    dpi: int = 200,
    plot_title: str | None = None,
    x_together: float | None = None,
    x_range: tuple[int, int] | None = None,
):
    """Plot relative ranks of the incumbent over time."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # For now we always want it flat...
    row_length = 100

    ncols = len(subtitle_results)
    nrows = math.ceil(ncols / row_length)
    figsize = (ncols * 4, nrows * 3)
    legend_ncol = len(algorithms)

    fig, _axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs: list[plt.Axes] = list(_axs.flatten())

    for col, ((subtitle, results), ax) in enumerate(zip(subtitle_results.items(), axs)):
        _x_range: tuple[int, int]
        if x_range is None:
            xmin = min(getattr(r, xaxis) for r in results.iter_results())
            xmax = max(getattr(r, xaxis) for r in results.iter_results())
            _x_range = (math.floor(xmin), math.ceil(xmax))
        else:
            _x_range = tuple(x_range)  # type: ignore

        left, right = _x_range
        xticks = get_xticks(_x_range)
        ymin, ymax = (0.8, len(algorithms))
        yticks = range(1, len(algorithms) + 1)
        center = (len(algorithms) + 1) / 2

        ax.set_title(subtitle, fontsize=18)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(X_LABEL.get(xaxis, xaxis), fontsize=18, color=(0, 0, 0, 0.69))
        ax.set_yticks(yticks)  # type: ignore
        ax.set_xlim(left=left, right=right)
        ax.set_xticks(xticks, xticks)  # type: ignore
        ax.tick_params(axis="both", which="major", labelsize=18, color=(0, 0, 0, 0.69))
        ax.grid(True, which="both", ls="-", alpha=0.8)

        if col == 0:
            ax.set_ylabel("Relative rank", fontsize=18, color=(0, 0, 0, 0.69))

        all_means, all_stds = results.ranks(xaxis=xaxis, yaxis=yaxis)

        for algorithm in algorithms:
            means: pd.Series = all_means[algorithm]  # type: ignore
            stds: pd.Series = all_stds[algorithm]  # type: ignore

            # If x_together is specified, we want to shave off
            # everything in the x-axis before the x_together index
            # so that it lines up with the above
            if x_together is not None:
                means = means.loc[x_together:]  # type: ignore
                stds = stds.loc[x_together:]  # type: ignore
            elif x_together is None:
                # Otherwise, we just use whatever the xaxis cutoff is
                means = means.loc[left:]
                stds = stds.loc[left:]

            # Center everything
            means.loc[0] = center
            stds.loc[0] = 0

            means = means.sort_index(ascending=True)  # type: ignore
            stds = stds.sort_index(ascending=True)  # type: ignore
            assert means is not None
            assert stds is not None

            x = np.array(means.index.tolist(), dtype=float)
            y = np.array(means.tolist(), dtype=float)
            std = np.array(stds.tolist(), dtype=float)

            ax.step(
                x=x,
                y=y,
                color=COLOR_MARKER_DICT.get(algorithm, "black"),
                linewidth=1,
                where="post",
                label=ALGORITHMS.get(algorithm, algorithm),
            )
            ax.fill_between(
                x,
                y - std,  # type: ignore
                y + std,  # type: ignore
                color=COLOR_MARKER_DICT.get(algorithm, "black"),
                alpha=0.1,
                step="post",
            )

    sns.despine(fig)
    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        fontsize="xx-large",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=legend_ncol,
        frameon=True,
    )

    for item in legend.legendHandles:
        item.set_linewidth(2)

    if plot_title:
        fig.suptitle(plot_title)

    fig.tight_layout(pad=0, h_pad=0.5)
    print(f"Saving relative-rank to {filepath}")
    fig.savefig(filepath, bbox_inches="tight", dpi=dpi)


def plot_incumbent_traces(
    results: ExperimentResults,
    filepath: Path,
    dpi: int = 200,
    plot_default: bool = True,
    plot_optimum: bool = True,
    yaxis: Literal["loss", "max_fidelity_loss"] = "loss",
    xaxis: Literal[
        "cumulated_fidelity",
        "end_time_since_global_start",
    ] = "cumulated_fidelity",
    xaxis_label: str | None = None,
    yaxis_label: str | None = None,
    x_range: tuple[int, int] | None = None,
    with_markers: bool = False,
    dynamic_y_lim: bool = False,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy import stats

    benchmarks = results.benchmarks
    algorithms = results.algorithms
    bench_configs = results.benchmark_configs
    all_indices = pd.Index(results.indices(xaxis=xaxis, sort=False))

    # We only enable the option if the benchmark has these recorded
    plot_default = plot_default and any(
        c.prior_error is not None for c in bench_configs.values()
    )

    plot_optimum = plot_optimum and any(
        c.optimum is not None for c in bench_configs.values()
    )

    if len(benchmarks) == 6:
        nrows = 2
        ncols = 3
    else:
        nrows = np.ceil(len(benchmarks) / 4).astype(int)
        ncols = min(len(benchmarks), 4)

    legend_ncol = len(algorithms) + sum([plot_default, plot_optimum])
    figsize = (4 * ncols, 3 * nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = list(axs.flatten()) if isinstance(axs, np.ndarray) else [axs]

    for i, benchmark in enumerate(benchmarks):
        benchmark_config = results.benchmark_configs[benchmark]
        benchmark_results = results[benchmark]

        ax = axs[i]
        xlabel = xaxis_label if xaxis_label else X_LABEL.get(xaxis, xaxis)
        xlabel = xlabel if is_last_row(i, nrows, ncols) else None

        ylabel = yaxis_label if yaxis_label else Y_LABEL
        ylabel = ylabel if is_first_column(i, ncols) else None


        _x_range: tuple[int, int]
        if x_range is None:
            xmin = min(getattr(r, xaxis) for r in benchmark_results.iter_results())
            xmax = max(getattr(r, xaxis) for r in benchmark_results.iter_results())
            _x_range = (math.floor(xmin), math.ceil(xmax))
        else:
            _x_range = tuple(x_range)  # type: ignore

        left, right = _x_range

        # Now that we've plotted all algorithms for the benchmark,
        # we need to set some dynamic limits
        if dynamic_y_lim:
            y_values = [
                getattr(result, yaxis)
                for result in benchmark_results.iter_results()
                if left <= getattr(result, xaxis) <= right
            ]
            y_min, y_max = min(y_values), max(y_values)
            dy = abs(y_max - y_min)

            plot_offset = 0.10
            ax.set_ylim(y_min - dy * plot_offset, y_max + dy * plot_offset)
        else:
            ylims = Y_LIMITS.get(benchmark)
            if ylims is None:
                ax.set_ylim(auto=True)
            else:
                down, up = ylims
                ax.set_ylim(down, up)

        ax.set_xlim(left=left, right=right)

        xticks = get_xticks(_x_range)
        ax.set_xticks(xticks, xticks)

        ax.set_title(
            DATASETS.get(benchmark, benchmark),
            fontsize=18,
            color=BENCHMARK_COLORS.get(benchmark, "black"),
        )

        ax.set_xlabel(xlabel, fontsize=18, color=(0, 0, 0, 0.69))
        ax.set_ylabel(ylabel, fontsize=18, color=(0, 0, 0, 0.69))

        # Black with some alpha
        ax.tick_params(
            axis="both", which="major", labelsize=18, labelcolor=(0, 0, 0, 0.69)
        )
        ax.grid(True, which="both", ls="-", alpha=0.8)

        if plot_default and benchmark_config.prior_error is not None:
            # NOTE: In the case of MFH good where we have taken a prior close
            # to the optimum, and additionally add 0.25 noise at every iteration,
            # there is no predefined prior line we can meaningfully plot. Each
            # run will see a different prior. For consistency in the plots, we
            # have chosen to take the mean line of RS+Prior as a proxy to the
            # averaged prior, as RS+Prior will always sample the prior first.
            mfh_good_prior_benchmarks = [
                "mfh3_good_prior-good",
                "mfh3_terrible_prior-good",
                "mfh6_good_prior-good",
                "mfh6_terrible_prior-good",
            ]
            if (
                "random_search_prior" in algorithms
                and benchmark in mfh_good_prior_benchmarks
            ):
                random_search_results = benchmark_results["random_search_prior"]
                values = random_search_results.df(index=xaxis, values=yaxis)
                prior_error = values.iloc[0].mean(axis=0)
            elif (
                "hyperband_prior" in algorithms
                and benchmark in mfh_good_prior_benchmarks
            ):
                hyperband_prior_results = benchmark_results["hyperband_prior"]
                values = hyperband_prior_results.df(index=xaxis, values=yaxis)
                prior_error = values.iloc[0].mean(axis=0)

            else:
                prior_error = benchmark_config.prior_error

            ax.axhline(
                prior_error,
                color="black",
                linestyle=":",
                linewidth=1.0,
                dashes=(5, 10),
                label="Mode",
            )

        if plot_optimum and benchmark_config.optimum is not None:
            # plot only if the optimum score is better than the first incumbent plotted
            ax.axhline(
                benchmark_config.optimum,
                color="black",
                linestyle="-.",
                linewidth=1.2,
                label="Optimum",
            )

        for algorithm in algorithms:
            print("-" * 50)
            print(f"Benchmark: {benchmark} | Algorithm: {algorithm}")
            print("-" * 50)
            df = benchmark_results[algorithm].df(index=xaxis, values=yaxis)
            assert isinstance(df, pd.DataFrame)

            missing_indices = all_indices.difference(df.index)
            if missing_indices is not None:
                for missing_i in missing_indices:
                    df.loc[missing_i] = np.nan

            df = df.sort_index(ascending=True)
            assert df is not None

            df = df.fillna(method="ffill", axis=0)

            x = df.index
            y_mean = df.mean(axis=1).values
            std_error = stats.sem(df.values, axis=1)

            # Slightly smaller marker than deafult
            MARKERSIZE = 4

            ax.step(
                x,
                y_mean,
                label=ALGORITHMS.get(algorithm, algorithm),
                color=COLOR_MARKER_DICT.get(algorithm, "black"),
                linestyle="-",
                linewidth=1,
                marker=CUSTOM_MARKERS.get(algorithm) if with_markers else None,
                markersize=MARKERSIZE,
                where="post",
            )
            ax.fill_between(
                x,
                y_mean - std_error,
                y_mean + std_error,
                color=COLOR_MARKER_DICT.get(algorithm, "black"),
                alpha=0.1,
                step="post",
            )

    bbox_y_mapping = {1: -0.25, 2: -0.11, 3: -0.07, 4: -0.05, 5: -0.04}
    reorganize_legend(
        fig=fig,
        axs=axs,
        to_front=["Mode", "Optimum"],
        bbox_to_anchor=(0.5, bbox_y_mapping[nrows]),
        ncol=legend_ncol,
    )

    sns.despine(fig)
    fig.tight_layout(pad=0, h_pad=0.5)

    print(f"Saving incumbent trace to {filepath}")
    fig.savefig(filepath, bbox_inches="tight", dpi=dpi)

def plot_single_incumbent_trace(
    results: ExperimentResults,
    filepath: Path,
    title: str,
    rr_results: ExperimentResults,
    rr_plot_title: list[str],
    dpi: int = 200,
    plot_default: bool = True,
    yaxis: Literal["loss", "max_fidelity_loss"] = "loss",
    xaxis: Literal[
        "cumulated_fidelity",
        "end_time_since_global_start",
    ] = "cumulated_fidelity",
    xaxis_label: str | None = None,
    yaxis_label: str | None = None,
    x_range: tuple[int, int] | None = None,
    x_together: float | None = None,
    with_markers: bool = False,
    dynamic_y_lim: bool = False,
    y_range: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = (6.4, 4.8)
):
    if len(results.benchmarks) > 1:
        raise ValueError("Only meant for plotting a single benchmark")

    if y_range and dynamic_y_lim:
        raise ValueError("Only one of `y_range` and `dynamic_y_lim`")

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy import stats

    benchmark = results.benchmarks[0]
    benchmark_results = results[benchmark]
    algorithms = results.algorithms
    benchmark_config = results.benchmark_configs[benchmark]
    all_indices = pd.Index(results.indices(xaxis=xaxis, sort=False))

    # We only enable the option if the benchmark has these recorded
    plot_default = plot_default and benchmark_config.prior_error is not None

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # Plot incumbent first
    ax = axes[0]
    assert isinstance(ax, plt.Axes)

    xlabel = xaxis_label if xaxis_label else X_LABEL.get(xaxis, xaxis)
    ylabel = yaxis_label if yaxis_label else Y_LABEL


    _x_range: tuple[int, int]
    if x_range is None:
        xmin = min(getattr(r, xaxis) for r in benchmark_results.iter_results())
        xmax = max(getattr(r, xaxis) for r in benchmark_results.iter_results())
        _x_range = (math.floor(xmin), math.ceil(xmax))
    else:
        _x_range = tuple(x_range)  # type: ignore

    left, right = _x_range

    # Now that we've plotted all algorithms for the benchmark,
    # we need to set some dynamic limits
    if y_range:
        y_min, y_max = y_range
        ax.set_ylim(y_min, y_max)
    elif dynamic_y_lim:
        y_values = [
            getattr(result, yaxis)
            for result in benchmark_results.iter_results()
            if left <= getattr(result, xaxis) <= right
        ]
        y_min, y_max = min(y_values), max(y_values)
        dy = abs(y_max - y_min)

        plot_offset = 0.10
        ax.set_ylim(y_min - dy * plot_offset, y_max + dy * plot_offset)
    else:
        ylims = Y_LIMITS.get(benchmark)
        if ylims is None:
            ax.set_ylim(auto=True)
        else:
            down, up = ylims
            ax.set_ylim(down, up)

    ax.set_xlim(left=left, right=right)

    FONTSIZE = 22
    xticks = get_xticks(_x_range)
    ax.set_xticks(xticks, xticks) # type: ignore

    ax.set_title(title, fontsize=FONTSIZE)

    ax.set_xlabel(xlabel, fontsize=FONTSIZE, color=(0, 0, 0, 0.69))
    ax.set_ylabel(ylabel, fontsize=FONTSIZE, color=(0, 0, 0, 0.69))

    # Black with some alpha
    ax.tick_params(
        axis="both", which="major", labelsize=FONTSIZE, labelcolor=(0, 0, 0, 0.69)
    )
    ax.grid(True, which="both", ls="-", alpha=0.8)

    if plot_default and benchmark_config.prior_error is not None:
        ax.axhline(
            benchmark_config.prior_error,  # type: ignore
            color="black",
            linestyle=":",
            linewidth=2.0,
            dashes=(5, 10),
            label="Mode",
        )

    # Slightly smaller marker than deafult
    MARKERSIZE = 6

    for algorithm in algorithms:
        print("-" * 50)
        print(f"Benchmark: {benchmark} | Algorithm: {algorithm}")
        print("-" * 50)
        df = benchmark_results[algorithm].df(index=xaxis, values=yaxis)
        assert isinstance(df, pd.DataFrame)

        missing_indices = all_indices.difference(df.index)
        if missing_indices is not None:
            for missing_i in missing_indices:
                df.loc[missing_i] = np.nan

        df = df.sort_index(ascending=True)
        assert df is not None

        df = df.fillna(method="ffill", axis=0)

        x = df.index
        y_mean = df.mean(axis=1).values
        std_error = stats.sem(df.values, axis=1)

        ax.step(
            x,
            y_mean,
            label=ALGORITHMS.get(algorithm, algorithm),
            color=COLOR_MARKER_DICT.get(algorithm, "black"),
            linestyle="-",
            linewidth=2,
            marker=CUSTOM_MARKERS.get(algorithm) if with_markers else None,
            markersize=MARKERSIZE,
            where="post",
        )
        ax.fill_between(
            x,
            y_mean - std_error,
            y_mean + std_error,
            color=COLOR_MARKER_DICT.get(algorithm, "black"),
            alpha=0.1,
            step="post",
        )

    bbox_y_mapping = {1: -0.20, 2: -0.11, 3: -0.07, 4: -0.05, 5: -0.04}
    legend_ncol = len(algorithms) + (1 if plot_default else 0)
    reorganize_legend(
        fig=fig,
        axs=ax,
        to_front=["Mode"],
        bbox_to_anchor=(0.5, bbox_y_mapping[1]),
        ncol=legend_ncol,
        fontsize=18,
    )

    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{int(y * 100)}" for y in yticks])

    # Plot the relative rankings
    ax = axes[1]

    _x_range: tuple[int, int]
    if x_range is None:
        xmin = min(getattr(r, xaxis) for r in results.iter_results())
        xmax = max(getattr(r, xaxis) for r in results.iter_results())
        _x_range = (math.floor(xmin), math.ceil(xmax))
    else:
        _x_range = tuple(x_range)  # type: ignore

    left, right = _x_range
    # HARDCODE:
    left = 0

    xticks = get_xticks(_x_range)
    yticks = range(1, len(algorithms) + 1)
    center = (len(algorithms) + 1) / 2

    ax.set_title(rr_plot_title, fontsize=FONTSIZE)
    ax.set_xlabel(X_LABEL.get(xaxis, xaxis), fontsize=FONTSIZE, color=(0, 0, 0, 0.69))
    ax.set_yticks(yticks)  # type: ignore
    ax.set_xlim(left=left, right=right)
    ax.set_ylim(1, len(algorithms))
    ax.set_xticks(xticks, xticks)  # type: ignore
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE, color=(0, 0, 0, 0.69))
    ax.grid(True, which="both", ls="-", alpha=0.8)
    ax.set_ylabel("Relative rank", fontsize=FONTSIZE, color=(0, 0, 0, 0.69))

    all_means, all_stds = rr_results.ranks(xaxis=xaxis, yaxis=yaxis)

    for algorithm in algorithms:
        means: pd.Series = all_means[algorithm]  # type: ignore
        stds: pd.Series = all_stds[algorithm]  # type: ignore

        # If x_together is specified, we want to shave off
        # everything in the x-axis before the x_together index
        # so that it lines up with the above
        if x_together is not None:
            means = means.loc[x_together:]  # type: ignore
            stds = stds.loc[x_together:]  # type: ignore
        elif x_together is None:
            # Otherwise, we just use whatever the xaxis cutoff is
            means = means.loc[left:]
            stds = stds.loc[left:]

        # Center everything
        means.loc[0] = center
        stds.loc[0] = 0

        means = means.sort_index(ascending=True)  # type: ignore
        stds = stds.sort_index(ascending=True)  # type: ignore
        assert means is not None
        assert stds is not None

        x = np.array(means.index.tolist(), dtype=float)
        y = np.array(means.tolist(), dtype=float)
        std = np.array(stds.tolist(), dtype=float)

        ax.step(
            x=x,
            y=y,
            color=COLOR_MARKER_DICT.get(algorithm, "black"),
            linewidth=2,
            where="post",
            # label=ALGORITHMS.get(algorithm, algorithm),  # Handled by incumbents
            #marker=CUSTOM_MARKERS.get(algorithm) if with_markers else None,
            #markersize=MARKERSIZE,
        )
        ax.fill_between(
            x,
            y - std,  # type: ignore
            y + std,  # type: ignore
            color=COLOR_MARKER_DICT.get(algorithm, "black"),
            alpha=0.1,
            step="post",
        )

    sns.despine(fig)
    fig.tight_layout(pad=0, h_pad=0.5)

    print(f"Saving single-inc-rr-trace to {filepath}")
    fig.savefig(filepath, bbox_inches="tight", dpi=dpi)

def tablify(
    results: ExperimentResults,
    filepath: Path,
    xs: list[int],
    prefix: str,
    *,
    yaxis: Literal["loss", "max_fidelity_loss"] = "loss",
) -> None:
    import pandas as pd

    n_algorithms = len(results.algorithms)
    n_budgets = len(xs)

    prior_order = ["good", "medium", "at25", "bad"]
    means, stds = results.table_results(xs=xs, yaxis=yaxis, sort_order=prior_order)

    # We'll just insert results into here later
    final_table = means.copy()

    for budget in xs:
        budget_means = means[budget]
        budget_stds = stds[budget]
        assert budget_means is not None
        assert budget_stds is not None

        str_version = (
            budget_means.round(3).astype(str) + "\\pm" + budget_stds.round(3).astype(str)
        )
        idx_min = budget_means.idxmin(axis=1)
        assert isinstance(idx_min, pd.Series)
        for i, algo in idx_min.items():
            str_version.loc[i, algo] = "\\bm{" + str_version.loc[i, algo] + "}"
        final_table[budget] = "$" + str_version + "$"

    # Rename benchmark names
    final_table.rename(index=BENCH_TABLE_NAMES, inplace=True)

    # Make sure to escape `_`
    final_table.rename(index=lambda bench: bench.replace("_", "\\_"), inplace=True)

    # Rename the budget top level columns
    final_table.rename(columns=lambda budget: f"{int(budget)}x", level=0, inplace=True)

    # Rename the algorithms
    final_table.rename(columns=ALGORITHMS, level=1, inplace=True)

    table_str = final_table.to_latex(
        escape=False,
        bold_rows=True,
        column_format="l | " + " | ".join(["c" * n_algorithms] * n_budgets),
        multicolumn_format="c",
    )  # type: ignore

    latex_str_header = "\n".join(
        [
            r"\begin{table}",
            r"\caption{\protect\input{captions/" + f"{prefix}-table-{yaxis}" + r"}}",
            r"\label{table:" + f"{prefix}-table-{yaxis}" + r"}",
            r"\begin{center}",
            r"\scalebox{0.55}{",
            r"\centering",
        ]
    )
    latex_str_footer = "\n".join(
        [
            r"}",
            r"\end{center}",
            r"\end{table}",
        ]
    )

    assert table_str is not None
    table_str = latex_str_header + table_str + latex_str_footer

    print(f"Writing table to {filepath}")
    with filepath.open("w") as f:
        f.write(table_str)


def main(
    experiment_group: str,
    prefix: str,
    algorithms: list[str] | None = None,
    incumbent_trace_benchmarks: dict[str, list[str]] | None = None,
    base_path: Path | None = DEFAULT_BASE_PATH,
    relative_rankings: dict[str, dict[str, list[str]]] | None = None,
    table_xs: list[int] | None = None,
    table_benchmarks: list[str] | None = None,
    plot_default: bool = True,
    plot_optimum: bool = True,
    dynamic_y_lim: bool = False,
    x_range_it: tuple[int, int] | None = None,
    x_range_rr: tuple[int, int] | None = None,
    x_together_rr: float | None = None,
    with_markers: bool = False,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    extension: str = "png",
    dpi: int = 200,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC_PARAMS)

    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    if algorithms is None:
        raise ValueError("Must specify --algorithms")

    plot_dir = base_path / "plots" / experiment_group
    xaxis = "cumulated_fidelity"
    yaxes = ["loss", "max_fidelity_loss"]

    CACHE = base_path / "results" / experiment_group / ".plot_cache.pkl"
    if not CACHE.exists():
        raise RuntimeError(f"No cache found at {CACHE}, run `--collect` first")

    print("-" * 50)
    print(f"Using cache at {CACHE}")
    print("-" * 50)
    with CACHE.open("rb") as f:
        results = pickle.load(f)

    # Implies it uses as many as cores available
    executor = ProcessPoolExecutor(max_workers=None)
    futures: list[Future] = []
    path_lookup: dict[Future, Path] = {}

    with executor:
        if incumbent_trace_benchmarks is not None:
            for yaxis in yaxes:
                for plot_title, _benches in incumbent_trace_benchmarks.items():
                    _plot_title = plot_title.lstrip().rstrip().replace(" ", "-")
                    _filename = f"{prefix}-{_plot_title}-{yaxis}.{extension}"
                    filepath = plot_dir / "incumbent_traces" / yaxis / _filename
                    filepath.parent.mkdir(parents=True, exist_ok=True)

                    kwargs = {
                        "results": results.select(
                            benchmarks=_benches, algorithms=algorithms
                        ),
                        "filepath": filepath,
                        "dpi": dpi,
                        "plot_default": plot_default,
                        "plot_optimum": plot_optimum,
                        "yaxis": yaxis,  # type: ignore
                        "xaxis": xaxis,  # type: ignore
                        "x_range": x_range_it,
                        "xaxis_label": x_axis_label,
                        "yaxis_label": y_axis_label,
                        "with_markers": with_markers,
                        "dynamic_y_lim": dynamic_y_lim,
                    }
                    func = with_traceback(plot_incumbent_traces)
                    future = executor.submit(func, **kwargs)
                    futures.append(future)
                    path_lookup[future] = filepath

        # Relative ranking plots
        if relative_rankings is not None:
            for yaxis in yaxes:
                for plot_title, plot_benchmarks in relative_rankings.items():
                    _plot_title = plot_title.lstrip().rstrip().replace(" ", "-")
                    _filename = f"{prefix}-{_plot_title}-{yaxis}.{extension}"
                    filepath = plot_dir / "relative-rankings" / yaxis / _filename
                    filepath.parent.mkdir(parents=True, exist_ok=True)

                    kwargs = {
                        "algorithms": algorithms,
                        "filepath": filepath,
                        "dpi": dpi,
                        "subtitle_results": {
                            sub_title: results.select(
                                benchmarks=_benches, algorithms=algorithms
                            )
                            for sub_title, _benches in plot_benchmarks.items()
                        },
                        "yaxis": yaxis,
                        "xaxis": xaxis,
                        "x_range": x_range_rr,
                        "x_together": x_together_rr,
                    }
                    func = with_traceback(plot_relative_ranks)
                    future = executor.submit(func, **kwargs)
                    futures.append(future)
                    path_lookup[future] = filepath

        if table_benchmarks is not None:
            assert table_xs is not None
            for yaxis in yaxes:
                _filename = f"{prefix}-table-{yaxis}.tex"
                filepath = plot_dir / "tables" / yaxis / _filename
                filepath.parent.mkdir(parents=True, exist_ok=True)

                kwargs = {
                    "results": results.select(
                        algorithms=algorithms,
                        benchmarks=table_benchmarks,
                    ),
                    "filepath": filepath,
                    "xs": table_xs,
                    "prefix": prefix,
                    "yaxis": yaxis,  # type: ignore
                }
                func = with_traceback(tablify)
                future = executor.submit(func, **kwargs)
                futures.append(future)
                path_lookup[future] = filepath

        # Should wait until all futures are done here but we explicitly do so anyways
        wait_futures(futures, return_when="ALL_COMPLETED")

    # If any errors occured during plotting, make sure print them
    for future in futures:
        exception = future.exception()
        if exception:
            print(f"Future raised an exception: {exception}")
            print(f"No plot was saved at {path_lookup[future]}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="mf-prior-exp plotting")

    parser.add_argument("--prefix", type=str, default=None)

    parser.add_argument("--collect", action="store_true")
    parser.add_argument(
        "--collect-ignore-benchmarks", type=str, nargs="+", default=None, required=False
    )
    parser.add_argument(
        "--collect-ignore-algorithms", type=str, nargs="+", default=None, required=False
    )
    parser.add_argument(
        "--collect-ignore-seeds", type=int, nargs="+", default=None, required=False
    )
    parser.add_argument("--collect-ignore-missing", action="store_true")

    parser.add_argument("--experiment_group", type=str, required=True)
    parser.add_argument("--algorithms", nargs="+", type=str, default=None)
    parser.add_argument("--benchmarks", nargs="+", type=str, default=None)

    parser.add_argument(
        "--incumbent_traces",
        type=json.loads,
        default=None,
        required=False,
        help=(
            "Expects a json dict like:\n"
            "{\n"
            "   'plot_title1': ['benchmark1', 'benchmark2', ...] },\n"
            "   'plot_title2': ['benchmark3', 'benchmark4', ...] },\n"
            "   ...,\n"
            "}"
        ),
    )
    parser.add_argument(
        "--relative_rankings",
        type=json.loads,
        default=None,
        required=False,
        help=(
            "Expects a json dict like:\n"
            "{\n"
            "   'plot_title1': {'subtitle1': ['benchmark', ...], 'subtitle2': ['benchmark', ...]} },\n"
            "   'plot_title2': {'subtitle1': ['benchmark', ...], 'subtitle2': ['benchmark', ...]} },\n"
            "   ...,\n"
            "}"
        ),
    )
    parser.add_argument(
        "--table_benchmarks",
        type=str,
        nargs="+",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--table_xs",
        type=float,
        nargs="+",
        default=None,
        required=False,
    )

    parser.add_argument("--base_path", type=Path, default=None)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--budget", nargs="+", type=float, default=None)
    parser.add_argument("--x_range_it", nargs=2, type=float, default=None)
    parser.add_argument("--x_range_rr", nargs=2, type=float, default=None)
    parser.add_argument("--x_together_rr", type=float, default=None)
    parser.add_argument("--with_markers", action="store_true")
    parser.add_argument("--x_axis_label", type=str, default=None)
    parser.add_argument("--y_axis_label", type=str, default=None)

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--ext", type=str, choices=["pdf", "png"], default="png")
    parser.add_argument("--plot_default", action="store_true")
    parser.add_argument("--plot_optimum", action="store_true")
    parser.add_argument("--dynamic_y_lim", action="store_true")
    parser.add_argument("--parallel", action="store_true")

    parser.add_argument("--single_inc_plot", action="store_true")
    parser.add_argument("--single_inc_y_range", type=float, nargs=2, default=None)
    parser.add_argument("--single_inc_figsize", type=float, nargs=2, default=None)
    parser.add_argument("--single_inc_plot_title", type=str)
    parser.add_argument("--single_inc_benchmark", type=str)
    parser.add_argument("--single_inc_y_log", action="store_true")
    parser.add_argument("--single_inc_rr_benchmarks", type=str, nargs="*")
    parser.add_argument("--single_inc_rr_plot_title", type=str, default=None)

    parser.add_argument("--regret_plot_title", type=str, default="regret_plot_default_name")

    args = parser.parse_args()

    if args.x_together_rr and args.x_range_rr and args.x_together_rr < args.x_range_rr[0]:
        raise ValueError("--x_together must be larger than --x_range[0]")

    if args.budget:
        raise ValueError("CD plots (which use --budget) not supported yet")

    if args.single_inc_plot:
        assert args.single_inc_benchmark is not None
        assert args.single_inc_plot_title is not None
        assert args.single_inc_rr_benchmarks is not None
        assert args.single_inc_rr_plot_title is not None

    return args


def collect(
    experiment_group: str,
    base_path: Path,
    n_workers: int,
    parallel: bool = True,
    ignore_missing: bool = False,
    ignore_benchmarks: set[str] | None = None,
    ignore_seeds: set[int] | None = None,
    ignore_algorithms: set[str] | None = None,
    save_file: bool = True,
) -> ExperimentResults:
    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    CACHE = base_path / "results" / experiment_group / ".plot_cache.pkl"
    CACHE.parent.mkdir(exist_ok=True, parents=True)

    # Fetch the results we need
    starttime = time.time()

    xaxis = "cumulated_fidelity"

    print(f"[{now()}] Processing ...")
    all_benchmarks, all_algorithms, all_seeds = all_possibilities(
        experiment_group,
        base_path,
        ignore_benchmarks=ignore_benchmarks,
        ignore_seeds=ignore_seeds,
        ignore_algorithms=ignore_algorithms,
    )
    results = fetch_results(
        experiment_group=experiment_group,
        benchmarks=list(all_benchmarks),
        algorithms=list(all_algorithms),
        seeds=sorted(all_seeds),
        base_path=base_path,  # Base path of the repo
        parallel=parallel,  # Whether to process in parallel
        n_workers=n_workers,  # Flag to indicate if it was a parallel setup
        continuations=True,  # Continue on fidelities from configurations
        cumulate_fidelities=True,  # Accumulate fidelities in the indices
        xaxis=xaxis,  # The x-axis to use
        rescale=False,
        rescale_xaxis="max_fidelity",  # We always rescale the xaxis by max_fidelity
        incumbent_value="loss",  # The incumbent is deteremined by the loss
        incumbents_only=True,  # We only want incumbent traces in our results
        ignore_missing=ignore_missing,
    )
    if save_file:
        with CACHE.open("wb") as f:
            pickle.dump(results, f)

    print(f"[{now()}] Done! Duration {time.time() - starttime:.3f}...")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    print("Plotting with args:")
    print(args)
    if args.collect:
        ignore_benchmarks = (
            set(args.collect_ignore_benchmarks)
            if args.collect_ignore_benchmarks
            else None
        )
        ignore_algorithms = (
            set(args.collect_ignore_algorithms)
            if args.collect_ignore_algorithms
            else None
        )
        ignore_seeds = (
            set(args.collect_ignore_seeds) if args.collect_ignore_seeds else None
        )
        collect(
            experiment_group=args.experiment_group,
            base_path=args.base_path,
            n_workers=args.n_workers,
            parallel=args.parallel,
            ignore_missing=args.collect_ignore_missing,
            ignore_benchmarks=ignore_benchmarks,
            ignore_algorithms=ignore_algorithms,
            ignore_seeds=ignore_seeds,
        )
    elif args.single_inc_plot:
        # TODO
        import matplotlib.pyplot as plt
        
        plt.rcParams.update(RC_PARAMS)
        base_path = args.base_path

        if base_path is None:
            base_path = DEFAULT_BASE_PATH

        if args.algorithms is None:
            raise ValueError("Must specify --algorithms")

        assert args.single_inc_benchmark is not None
        single_inc_benchmark = args.single_inc_benchmark

        plot_dir = base_path / "plots" / args.experiment_group
        xaxis = "cumulated_fidelity"
        yaxes = ["loss", "max_fidelity_loss"]

        CACHE = base_path / "results" / args.experiment_group / ".plot_cache.pkl"
        if not CACHE.exists():
            raise RuntimeError(f"No cache found at {CACHE}, run `--collect` first")

        print("-" * 50)
        print(f"Using cache at {CACHE}")
        print("-" * 50)
        with CACHE.open("rb") as f:
            results = pickle.load(f)

        yaxis = "max_fidelity_loss"
        _plot_title = args.single_inc_plot_title.lstrip().rstrip().replace(" ", "-")
        _filename = f"{args.prefix}-{_plot_title}-{yaxis}.{args.ext}"
        filepath = plot_dir / "single_inc" / yaxis / _filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        plot_single_incumbent_trace(
            results=results.select(benchmarks=[single_inc_benchmark], algorithms=args.algorithms),
            filepath=filepath,
            dpi=args.dpi,
            plot_default=args.plot_default,
            yaxis=yaxis,  # type: ignore
            xaxis=xaxis,
            y_range=args.single_inc_y_range,
            x_range=args.x_range_it,
            yaxis_label=args.y_axis_label,
            with_markers=args.with_markers,
            dynamic_y_lim=args.dynamic_y_lim,
            figsize=args.single_inc_figsize,
            title=args.single_inc_plot_title,
            x_together=args.x_together_rr,
            rr_results=results.select(benchmarks=args.single_inc_rr_benchmarks, algorithms=args.algorithms),
            rr_plot_title=args.single_inc_rr_plot_title,
        )
    elif args.benchmarks is not None:
        # collect all results (without saving) into a hierarchical dict
        # {benchmarks: {algorithms: {seeds: trace}}
        _all_benchmarks, _all_algorithms, _all_seeds = all_possibilities(
            args.experiment_group,
            args.base_path
        )
        ignore_benchmarks = set(_all_benchmarks) - set(args.benchmarks)
        ignore_algorithms = set(_all_algorithms) - set(args.algorithms)
        ignore_seeds = (
            set(args.collect_ignore_seeds) if args.collect_ignore_seeds else None
        )
        results = collect(
            experiment_group=args.experiment_group,
            base_path=args.base_path,
            n_workers=args.n_workers,
            parallel=args.parallel,
            ignore_missing=args.collect_ignore_missing,
            ignore_benchmarks=ignore_benchmarks,
            ignore_algorithms=ignore_algorithms,
            ignore_seeds=ignore_seeds,
            save_file=False,  # prevents overwriting the plot cache with a subset ofruns
        )
        # find minimum error per benchmark
        bench_global_min = dict()
        for bench, bench_dict in results.results.items():
            _min = float("inf")
            _max = float("-inf")
            for algo, algo_dict in bench_dict.items():
                for seed, trace in algo_dict.items():
                    _min = min(_min, trace.df.min_valid_seen.values.min())
                    _max = max(_max, trace.df.min_valid_seen.values.max())
            bench_global_min[bench] = (_min, _max)
        
        print("Global minima for this experiment: ")
        print(bench_global_min)
        
        import pandas as pd
        norm = lambda x, l, u: (x - l) / (u - l)
        bench_data = dict()
        for bench, bench_dict in results.results.items():
            _min, _max = bench_global_min[bench]
            bench_data[bench] = dict()
            for algo, algo_dict in bench_dict.items():  
                run_data = pd.DataFrame()
                for seed, trace in algo_dict.items():
                    run_data[seed] = pd.Series(
                        norm(trace.df.min_valid_seen.values, _min, _max),
                        index=trace.df.cumulated_fidelity
                    )
                run_data = run_data.fillna(method="ffill")
                bench_data[bench][algo] = dict(
                    index=run_data.index.values,
                    mean=run_data.values.mean(axis=1),
                    # calculating standard error of mean
                    std=run_data.values.std(axis=1) / np.sqrt(len(run_data.columns)),
                )  

        num_plots = len(bench_data)
        if num_plots <= 4:
            nrows = 1
            ncols = num_plots
        elif num_plots <= 8:
            nrows = 2
            ncols = np.ceil(num_plots // nrows).astype(int)
        elif num_plots <= 12:
            nrows = 3
            ncols = np.ceil(num_plots // nrows).astype(int)
        else:
            raise ValueError("Too many plots")

        from matplotlib import pyplot as plt

        plt.clf()
        fig, ax = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), sharey=True
        )
        bench_names = list(bench_data.keys())
        
        for row in range(nrows):
            for col in range(ncols):
                idx = row * ncols + col
                axes = ax[row, col] if nrows > 1 else (ax if nrows == 1 and ncols == 1 else ax[col])
                for algo, algo_dict in bench_data[bench_names[idx]].items():
                    axes.step(x=algo_dict["index"], y=algo_dict["mean"], label=algo)
                    axes.fill_between(
                        x=algo_dict["index"], 
                        y1=algo_dict["mean"] - algo_dict["std"], 
                        y2=algo_dict["mean"] + algo_dict["std"], 
                        alpha=0.3,
                        step="post",
                    )
                axes.set_title(bench_names[idx])
                axes.set_yscale("log")
                axes.set_xscale("log")
                axes.legend()
        
        plot_dir = args.base_path / "plots" / args.experiment_group / "regret" / f"{args.regret_plot_title}.{args.ext}"
        plot_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving regret plot to {plot_dir}")        
        fig.savefig(plot_dir, bbox_inches="tight", dpi=150)

    else:
        main(
            experiment_group=args.experiment_group,
            algorithms=args.algorithms,
            incumbent_trace_benchmarks=args.incumbent_traces,
            prefix=args.prefix,
            base_path=args.base_path,
            relative_rankings=args.relative_rankings,
            x_range_it=args.x_range_it,
            x_range_rr=args.x_range_rr,
            x_together_rr=args.x_together_rr,
            with_markers=args.with_markers,
            x_axis_label=args.x_axis_label,
            y_axis_label=args.y_axis_label,
            plot_default=args.plot_default,
            plot_optimum=args.plot_optimum,
            extension=args.ext,
            dpi=args.dpi,
            dynamic_y_lim=args.dynamic_y_lim,
            table_xs=args.table_xs,
            table_benchmarks=args.table_benchmarks,
        )
