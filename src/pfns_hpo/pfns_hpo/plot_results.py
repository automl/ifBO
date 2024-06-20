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
import hickle as hkl

from copy import deepcopy
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from typing_extensions import Literal, List

from pfns_hpo.utils.plotting_utils import (
    calc_bounds_per_benchmark,
    get_aggregated_plot,
    get_rank_plot,
    get_per_benchmark_plot,
    normalize_and_calculate_regrets,
    MAX_N_COLS,
    reorder_for_aggregated_benchmark_plots,
    plt,
    group_run_dataframes,
    get_plot_styles,
PLOT_SIZE

)

from neps.status.status import get_run_summary_csv
from tueplots import bundles, figsizes, fontsizes


HERE = Path(__file__).parent.absolute()
DEFAULT_BASE_PATH = HERE.parent / "results"


def get_aggregated_plot_style(
        plot_data: dict,
        filename: str,
        log_x: bool = False,
        log_y: str = False,
        x_range: tuple = None,
        color_map: dict = None,
        marker_map: dict = None,
        marker_args: dict = None,
        label_map: dict = None,
        ax: plt.Axes = None
) -> None:
    """Plots a single plot aggregating performance of each algorithm across benchmarks."""
    plot_data = reorder_for_aggregated_benchmark_plots(plot_data)

    # processing data for plotting
    algo_perf = dict()  # nothing to do with https://arxiv.org/abs/2306.07179
    for algo, algo_data in plot_data.items():
        algo_perf[algo] = dict()
        seeds = []
        for seed, seed_data in algo_data.items():
            seeds.append(seed)
            to_delete = []

            if algo == "random_search": 
                # Special treatment for random search on PD1
                to_delete = []
                for namedataset, dataset in seed_data.items():
                    if dataset.cumsum_fidelity.min() >= 1000: 
                        to_delete.append(namedataset)
                seed_data = {k: v for k, v in seed_data.items() if k not in to_delete}

            # collecting the mean score across benchmarks
            algo_perf[algo][seed], _ = group_run_dataframes(list(seed_data.values()), x_range=x_range, nanmean=True if algo=="random_search" else False)
        # averaging score across seeds
        algo_perf[algo]["mean"], algo_perf[algo]["sem"] = (
            group_run_dataframes(list(algo_perf[algo].values()), x_range=x_range)
        )
        # removing seed keys
        _ = [algo_perf[algo].pop(_seed) for _seed in seeds]

    # Define a list of line styles and markers
    if color_map is None or marker_map is None:
        l_colors, l_line_styles, l_markers = get_plot_styles(algo_perf)
    else:
        l_colors = color_map
        l_markers = marker_map
    # l_colors, l_line_styles, l_markers = get_plot_styles(algo_perf)

    # Do the plotting
    algo_order = [(algo, algo_data["mean"].values[-1]) for algo, algo_data in algo_perf.items()]
    algo_order.sort(key=lambda x: x[1], reverse=True)

    for i, (algo, _) in enumerate(algo_order):
        algo_data = algo_perf[algo]
        ax.plot(
            algo_data["mean"].index.values,
            algo_data["mean"].values,
            color=l_colors[algo],
            # linestyle=l_line_styles[algo],
            marker=l_markers[algo],
            # markersize=6,
            markevery=max(1, int(len(algo_data["mean"].index.values) / 15)),
            # fillstyle='none',
            **marker_args,
            label=algo if algo not in label_map else label_map[algo],
            linewidth=1,
        )
        ax.fill_between(
            algo_data["mean"].index.values,
            algo_data["mean"].values - algo_data["sem"].values,
            algo_data["mean"].values + algo_data["sem"].values,
            facecolor=l_colors[algo],
            alpha=0.1,
            step="post"
        )
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel("Normalized regret")
    if log_x:
        ax.set_xscale("log")
    if x_range is not None:
        ax.set_xlim(*x_range)

    # Move the legend to the right side of the plot
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xlabel("Total epochs spent")


def get_rank_plot_style(plot_data: dict,
    filename: str,
    x_range: tuple=None,
    color_map: dict = None,
    marker_map: dict = None,
    marker_args: dict = None,
    label_map: dict = None,
    wallclock: bool = False,
    overhead: bool = False,
    ax: plt.Axes = None
) -> None:

    df_rank = pd.DataFrame()

    l_datasets = list(plot_data.keys())
    l_algos = plot_data[l_datasets[0]].keys()
    l_colors, l_line_styles, l_markers = get_plot_styles({algo: [] for algo in l_algos})

    aggregate_by = "cumsum_fidelity"

    for dataset in plot_data.keys():
        ldf = []

        for algo in l_algos:
            for seed in range(len(plot_data[dataset][algo])):
                df = plot_data[dataset][algo][seed][[aggregate_by, "inc_loss"]].set_index(aggregate_by, drop=True)
                df = df.loc[df.index.dropna()]  # removing NaNs in index
                df = df.loc[~df.index.duplicated(keep="first")]  # removing duplicate index
                df = df.reindex(np.arange(1, x_range[1] + 1), method="ffill").sort_index()
                df["model"] = algo
                df["seed"] = seed
                ldf.append(df)

        df = pd.concat(ldf, ignore_index=False).reset_index()
        df = df.pivot(columns="model", index=[aggregate_by, "seed"], values="inc_loss").rank(ascending=True, axis=1)
        df = df.melt(value_vars=l_algos, value_name="rank", ignore_index=False).reset_index()
        df["dataset"] = dataset
        df_rank = pd.concat([df_rank, df], axis=0, ignore_index=True)

    df_rank = df_rank.groupby([aggregate_by, "model"])["rank"].agg(['mean','sem'])
    df_rank = df_rank.reset_index().sort_values(aggregate_by, ascending=True)
    ordered_algos = df_rank[df_rank.loc[:, aggregate_by] == df_rank.loc[: ,aggregate_by].max()][["model", "mean"]].sort_values("mean", ascending=False).model.tolist()

    for i, algo in enumerate(ordered_algos):
        ax.plot(
            df_rank[df_rank.model == algo].loc[:, aggregate_by].values,
            df_rank[df_rank.model == algo]["mean"].values,
            color=color_map[algo],
            marker=marker_map[algo],
            markevery=50,
            **marker_args,
            label=label_map[algo],
            linewidth=1,
        )
        ax.fill_between(
            df_rank[df_rank.model == algo].loc[:, aggregate_by].values,
            df_rank[df_rank.model == algo]["mean"].values - df_rank[df_rank.model == algo]["sem"].values,
            df_rank[df_rank.model == algo]["mean"].values + df_rank[df_rank.model == algo]["sem"].values,
            facecolor=color_map[algo],
            alpha=0.1,
            step="post"
        )
    ax.set_ylim(0, len(l_algos))

    ax.set_xlabel("Total epochs spent")
    ax.set_ylabel("Rank")


def parse_args():
    parser = ArgumentParser(description="lc-pfn-hpo plotting")

    parser.add_argument("--basedir", type=str, default=None)

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Expects a list of algorithms, comma separated str. Needs at least one."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        required=False,
        help=(
            "If `None`, uses all seeds detected. "
            "If len(args.seeds) is 1, then interprets it as `range(args.seeds)`. "
            "If len(args.seeds) is 2, then interprets it as `range(*args.seeds)`. "
            "Anything else will throw an error."
        )
    )

    parser.add_argument("--filename", type=str, default=None, help="Name sans extension.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If True, requires that no runs are missing."
    )

    return parser.parse_args()


import seaborn as sns

DEFAULT_MARKER_KWARGS = dict(
    markersize=2,
    fillstyle="full",
    markeredgewidth=1,
    markerfacecolor="white",
)

# Colorblind
COLORS = sns.color_palette("bright") + sns.color_palette("colorblind")
MARKERS = ['o', 'v', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '^', '<', '>']

color_map = {"random_search": COLORS[7],
             "pfn-bopfn-broken-unisep-pi-random": COLORS[0],
             "pfn-bopfn-broken-expsep-pi-random": COLORS[9],
             "hyperband": COLORS[1],
             "asha": COLORS[5],
             "mf_ei_bo": COLORS[2],
             "dyhpo-neps-v2": COLORS[3],
             "dpl-neps-max": COLORS[4],
             "dyhpo-arlind": COLORS[3],
             "dpl-arlind": COLORS[4],
            #  "pfn-bopfn-broken-unisep-pi-random": COLORS[0],
            # Ablation acquisition function
            "pfn-bopfn-broken-unisep-pi": COLORS[19],
            "pfn-bopfn-broken-unisep-pi-max" : COLORS[18],
            "pfn-bopfn-broken-unisep-ei": COLORS[16],
            "pfn-bopfn-broken-unisep-pi-thresh-max" : COLORS[13],
            "pfn-bopfn-broken-unisep-pi-random-horizon" : COLORS[14],
            "pfn-bopfn-broken-unisep-ei-max": COLORS[12],
            # ablation surrogate
            "pfn-bopfn-broken-ablation-pow-pi-random": COLORS[12],
            "pfn-bopfn-broken-ablation-exp-pi-random": COLORS[14],
            "pfn-bopfn-broken-ablation-ilog-pi-random": COLORS[16],
            "pfn-bopfn-broken-ablation-hill-pi-random": COLORS[18],
            "pfn-bopfn-broken-ablation-nb-pow-pi-random": COLORS[12],
            "pfn-bopfn-broken-ablation-nb-exp-pi-random": COLORS[14],
            "pfn-bopfn-broken-ablation-nb-ilog-pi-random": COLORS[16],
            "pfn-bopfn-broken-ablation-nb-hill-pi-random": COLORS[18],
            "pfn-bopfn-broken-ablation-nb-comb-pi-random": COLORS[10],
            # ablation hps
            "pfn-bopfn-broken-no-hps-pi-random": COLORS[12],
}

marker_map = {"random_search": MARKERS[0],
              "pfn-bopfn-broken-unisep-pi-random": MARKERS[1],
              "pfn-bopfn-broken-expsep-pi-random": MARKERS[1],
              "hyperband": MARKERS[2],
              "asha": MARKERS[3],
              "mf_ei_bo": MARKERS[4],
              "dyhpo-neps-v2": MARKERS[5],
              "dpl-neps-max": MARKERS[6],
              "dyhpo-arlind": MARKERS[5],
              "dpl-arlind": MARKERS[6],
            #   "pfn-bopfn-broken-unisep-pi-random": MARKERS[1],
            # Ablation acquisition function
            "pfn-bopfn-broken-unisep-pi": MARKERS[1],
            "pfn-bopfn-broken-unisep-pi-max" : MARKERS[1],
            "pfn-bopfn-broken-unisep-ei": MARKERS[1],
            "pfn-bopfn-broken-unisep-pi-thresh-max" : MARKERS[1],
            "pfn-bopfn-broken-unisep-pi-random-horizon" : MARKERS[1],
            "pfn-bopfn-broken-unisep-ei-max": MARKERS[1],
            # Ablation surrogate
            "pfn-bopfn-broken-ablation-pow-pi-random": MARKERS[1],
            "pfn-bopfn-broken-ablation-exp-pi-random": MARKERS[1],
            "pfn-bopfn-broken-ablation-ilog-pi-random": MARKERS[1],
            "pfn-bopfn-broken-ablation-hill-pi-random": MARKERS[1],
            "pfn-bopfn-broken-ablation-nb-pow-pi-random": MARKERS[6],
            "pfn-bopfn-broken-ablation-nb-exp-pi-random": MARKERS[6],
            "pfn-bopfn-broken-ablation-nb-ilog-pi-random": MARKERS[6],
            "pfn-bopfn-broken-ablation-nb-hill-pi-random": MARKERS[6],
            "pfn-bopfn-broken-ablation-nb-comb-pi-random": MARKERS[6],
            # ablation hps
            "pfn-bopfn-broken-no-hps-pi-random": MARKERS[6],
}

label_map = {"random_search": "Random Search",
             "pfn-bopfn-broken-unisep-pi-random": "ifBO",  # "ICL-FT-BO",
             # "pfn-bopfn-broken-expsep-pi-random": "CL-FT-BO",
             "hyperband": "Hyperband",
             "asha": "ASHA",
             "mf_ei_bo": "Freeze-Thaw with GPs",
               "dyhpo-neps-v2": "DyHPO",
               "dpl-neps-max": "DPL",
             "dyhpo-arlind": "Original paper code",
             "dpl-arlind": "Original paper code",
            # dpl
            "dpl-neps-pi": "DPL (PI)",
            "dpl-neps-pi-random": "DPL (PI-random)",
            "dpl-neps-pi-max": "DPL (PI-max)",
            "dpl-neps-ei": "DPL (EI)",
            "dpl-neps-ei-random-horizon": "DPL (EI-random-horizon)",
            "dpl-neps-ei-max": "DPL (EI-max)",
            "dpl-neps-ei-random": "DPL (EI-random)",
            # Ablation acquisition function
            # "pfn-bopfn-broken-unisep-pi-random": "PI-random (ours)",
            "pfn-bopfn-broken-unisep-pi": "PI (one step)",
            "pfn-bopfn-broken-unisep-pi-max" : "PI (max)",
            "pfn-bopfn-broken-unisep-ei": "EI (one step)",
            "pfn-bopfn-broken-unisep-pi-thresh-max" : "PI (max, random-T)",
            "pfn-bopfn-broken-unisep-pi-random-horizon" : "PI (random horizon)",
            "pfn-bopfn-broken-unisep-ei-max": "EI (max)",
            # Ablation surrogate
            "pfn-bopfn-broken-ablation-pow-pi-random": "pow",
            "pfn-bopfn-broken-ablation-exp-pi-random": "exp",
            "pfn-bopfn-broken-ablation-ilog-pi-random": "ilog",
            "pfn-bopfn-broken-ablation-hill-pi-random": "hill",
            "pfn-bopfn-broken-ablation-nb-pow-pi-random": "pow (not broken)",
            "pfn-bopfn-broken-ablation-nb-exp-pi-random": "exp (not broken)",
            "pfn-bopfn-broken-ablation-nb-ilog-pi-random": "ilog (not broken)",
            "pfn-bopfn-broken-ablation-nb-hill-pi-random": "hill (not broken)",
            "pfn-bopfn-broken-ablation-nb-comb-pi-random": "comb (not broken)",
            # ablation hps
            "pfn-bopfn-broken-no-hps-pi-random": "ICL-FT-BO (no HPs)",
}

algorithms = ["random_search",
              "pfn-bopfn-broken-unisep-pi-random",
              "hyperband",
              "asha",
              "mf_ei_bo"]
neps_v = ["dyhpo-neps-v2",
          "dpl-neps-max"]
arlind = ["dyhpo-arlind",
          "dpl-arlind"]

name_benchmarks = {
    "lcbench": "LCBench",
    "pd1": "PD1",
    "taskset": "Taskset",
}

if __name__ == "__main__":
    args = parse_args()

    # Basic argument checks
    if args.seeds is not None:
        assert len(args.seeds) <= 2, "Invalid --seeds. Check --help."
    assert len(args.algorithms) > 0, "Invalid --algorithms. Check --help."

    print("Loading results ...")
    plot_data = hkl.load('allresults.hkl')
    bounds = hkl.load('bounds.hkl')

    plot_data = normalize_and_calculate_regrets(plot_data, bounds)

    plt.rcParams.update(figsizes.icml2024_full(nrows=2, ncols=3, height_to_width_ratio=.6))
    plt.rcParams.update(fontsizes.icml2024())

    fig, axes = plt.subplots(2, 3)

    for j, bench in enumerate(["lcbench", "pd1", "taskset"]):
        print(f"Plotting {bench}...")
        bench_data = {b: {a: plot_data[b][a] for a in args.algorithms} for b in plot_data if bench in b}
        get_aggregated_plot_style(
                bench_data.copy(),
                filename="aggregated" if args.filename is None else f"aggregated_{args.filename}",
                log_x=False,
                log_y=True,
                x_range=[0, 1000],
                color_map=color_map,
                marker_map=marker_map,
                marker_args=DEFAULT_MARKER_KWARGS,
                label_map=label_map,
                ax=axes[0, j],
        )
        axes[0, j].set_title(name_benchmarks[bench])

        get_rank_plot_style(
            bench_data.copy(),
            filename="rank" if args.filename is None else f"rank_{args.filename}",
            x_range=[0, 1000],
            color_map=color_map,
            marker_map=marker_map,
            marker_args=DEFAULT_MARKER_KWARGS,
            label_map=label_map,
            wallclock=False,
            overhead=False,
            ax=axes[1, j],
        )
        axes[1, j].set_title(name_benchmarks[bench])

    axes[0, 1].legend()
    hardcoded_legend_order = [1, 0, 4, 2, 5, 3,6]
    handles, labels = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].get_legend().remove()
    fig.legend([handles[_] for _ in hardcoded_legend_order], [labels[_] for _ in hardcoded_legend_order], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=7)
    fig.savefig(f"{args.filename}.pdf", bbox_inches="tight", dpi=600)
