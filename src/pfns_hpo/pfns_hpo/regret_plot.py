from __future__ import annotations

import json
import math
import pickle
import sys
import time
import traceback
from argparse import ArgumentParser, Namespace
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import wait as wait_futures
from pathlib import Path

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
    PLOT_SIZE,
)

from neps.status.status import get_run_summary_csv

# get_run_summary_csv(root_dir)


# from .plot_styles import (
#     ALGORITHMS,
#     BENCH_TABLE_NAMES,
#     BENCHMARK_COLORS,
#     CUSTOM_MARKERS,
#     COLOR_MARKER_DICT,
#     DATASETS,
#     RC_PARAMS,
#     X_LABEL,
#     Y_LABEL,
#     Y_LIMITS,
#     get_xticks,
# )
# from .plotting_types import ExperimentResults, all_possibilities, fetch_results


HERE = Path(__file__).parent.absolute()
DEFAULT_BASE_PATH = HERE.parent / "results"


# is_last_row = lambda idx, nrows, ncols: idx >= (nrows - 1) * ncols
# HERE = Path(__file__).parent.absolute()
# is_first_column = lambda idx, ncols: idx % ncols == 0

TUEPLOTS_SPECS ={
    "axes.labelsize": 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 20,
    'legend.fontsize': 20,
    'font.size': 20,
    'savefig.pad_inches': 0.025,
}

ANALYSIS_Y_LABEL = dict(
    mean_rank="Mean rank of top-3 partial learning curves",
    min_rank="Min. rank of top-3 partial learning curves",
    intersection_double="% of partial learning curves in top ranking set",
    intersection_p99="% of partial learning curves in top 99 percentile",
)


class with_traceback:

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            raise type(e)(f"{tb}\n{str(e)}") from e


def calculate_continuations(
    df: pd.DataFrame, fid_var: str = "result.info_dict.fidelity", inplace: bool = False
) -> pd.DataFrame:
    """Subtracts the previous recorded fidelity from each configuration."""
    _column = fid_var
    _df = df if inplace else df.copy()
    if "_" not in str(_df.Config_id.values[0]):
        # no continuations if single fidelity algorithm with no budget_id available
        return _df
    # retains the base IDs only
    _df.loc[:, "Config_id_base"] = _df["Config_id"].apply(lambda x: x.split('_')[0])
    # subtracts the previous fidelity value from the current value
    _df.loc[:, "fidelity_diff"] = _df.groupby("Config_id_base")[_column].diff()
    # fills only the values for the first fidelity which will be NaNs from the `.diff()`
    _df["fidelity_diff"].fillna(_df[_column], inplace=True)
    _df[_column] = _df["fidelity_diff"]
    _df.drop(columns=["fidelity_diff", "Config_id_base"], inplace=True)
    return _df


def _find_correct_path(path: Path, strict: bool=False) -> Path:
    if (path / "neps_root_directory").exists():
        path = path / "neps_root_directory"
        assert (path / "summary_csv").exists(), f"Run {path} not completed?"
        path = path / "summary_csv"
        assert (path / "config_data.csv").exists(), f"Run {path} not completed?"
        return path / "config_data.csv"
    if (path / "arlind_root_directory").exists():
        path = path / "arlind_root_directory"
        assert (path / "run_data.csv").exists(), f"Run {path} not completed?"
        return path / "run_data.csv"
    raise ValueError(f"Cannot find relevant subdirectory in {path}!")


def collect_a_run(
    path: Path, 
    continuations: bool = False, 
    wallclock: bool = False, 
    overhead: bool = False, 
    strict: bool = False, 
    analysis: bool=False
) -> pd.DataFrame:
    """Collects a single run for a benchmark-algorithm-seed combination."""
    assert path.exists(), f"{path} does not exist!"

    try:
        path = _find_correct_path(path)
    except AssertionError as e:
        if strict:
            raise Exception(repr(e))
        else:
            get_run_summary_csv(path / "neps_root_directory")
            path = _find_correct_path(path)

    df = pd.read_csv(path, float_precision="round_trip")
    if analysis:
        try:
            df_analysis = pd.read_csv(
                path.parent.parent / "analysis_data.csv", float_precision="round_trip", index_col=0
            )
            df = pd.concat([df, df_analysis], axis=1)
        except:
            pass

    # sort by time
    # NOTE: key assumption - only single worker runs
    time_cols = ["result.info_dict.start_time", "result.info_dict.end_time"]
    df.sort_values(by=time_cols, inplace=True, ignore_index=True)

    # calculate continuations
    if continuations:
        if "arlind_root_directory" in str(path):
            df.loc[:, "result.info_dict.fidelity"] = 1
        else:
            calculate_continuations(df, inplace=True)

    # adding cumulative fidelities for the x-axis
    df.loc[:, "cumsum_fidelity"] = df["result.info_dict.fidelity"].cumsum()
    # make cumulative fidelity the index of the sorted dataframe
    # df.set_index("cumsum_fidelity")
    df = df.set_index(df.cumsum_fidelity.values)

    # adding a column with the incumbent trace of the `loss`
    df.loc[:, "inc_loss"] = np.minimum.accumulate(df["result.loss"].values)

    # adding benchmark costs
    df.loc[:, "benchmark_costs"] = (
        df.loc[:, "metadata.time_end"] - df.loc[:, "metadata.time_sampled"]
    )
    # adding sampling_times
    df.loc[:, "sampling_times"] = df.loc[:, "metadata.time_sampled"].diff(-1).fillna(0).abs()
    # adding overhead time
    df.loc[:, "overhead"] = (
        df.loc[:, "sampling_times"] - df.loc[:, "benchmark_costs"]
    ).clip(0, 1e24)

    # adding wallclock time
    df.loc[:, "wallclock_without_overhead"] = df.loc[:, "result.info_dict.cost"]  #.cumsum()
    # adding wallclock time with overhead
    df.loc[:, "wallclock_with_overhead"] = (
        df.loc[:, "wallclock_without_overhead"] + df.loc[:, "overhead"]  #.cumsum()
    )

    # summing up overhead time
    df.loc[:, "overhead"] = df.loc[:, "overhead"]  # .cumsum()

    if (wallclock or overhead) and continuations and "arlind_root_directory" not in str(path):
        fid_variable = "result.info_dict.fidelity"
        if wallclock and overhead:
            fid_variable = "wallclock_with_overhead"
        elif wallclock and not overhead:
            fid_variable = "wallclock_without_overhead"
        elif not wallclock and overhead:
            fid_variable = "overhead"
        
        # if fid_variable in ["wallclock_with_overhead", "wallclock_without_overhead", "overhead"]:
        if fid_variable in ["wallclock_without_overhead"]:
            # calculate continuations
            calculate_continuations(df, fid_var=fid_variable, inplace=True)    
            # adding cumulative fidelities for the x-axis
            df.loc[:, "cumsum_fidelity"] = df[fid_variable].cumsum()
            # make cumulative fidelity the index of the sorted dataframe
            df = df.set_index(df.cumsum_fidelity.values)

    return df


def collect_all_seeds(
        path: Path,
        seeds: List[int] = None,
        continuations: bool = False,
        wallclock: bool = False, 
        overhead: bool = False, 
        strict: bool = False,
        analysis: bool = False,
) -> dict | pd.DataFrame:
    """Collects runs for at the seed level for a benchmark-algorithm."""
    assert path.exists(), f"{path} does not exist!"

    # collect all seeds run
    seed_dirs = list(path.iterdir())

    # function to filter seeds
    check_seeds = lambda p, seed_list: int(p.name.split('=')[-1]) in seed_list
    if seeds is None:
        pass
    else:
        _seeds = range(*seeds)  # handles both cases when len(seeds) is 1 or 2
        seed_dirs = [_dir for _dir in seed_dirs if check_seeds(_dir, _seeds)]

    # parallelized collection
    # NOTE: will use ALL available CPUs
    all_seed_runs = Parallel(n_jobs=-1)(delayed(
        lambda _dir: (
            int(_dir.name.split('=')[-1]), collect_a_run(
                _dir, continuations, wallclock, overhead, strict, analysis
            )
        )
    )(_dir) for _dir in seed_dirs)
    all_seed_runs = dict(all_seed_runs)
    # all_runs = pd.DataFrame.from_dict(all_runs)

    return all_seed_runs


def collect_algorithms(
        path: Path,
        benchmark: str,
        algorithms: List[str],
        seeds: List[int] = None,
        continuations: bool = False,
        wallclock: bool = False, 
        overhead: bool = False, 
        strict: bool = False,
        analysis: bool = False,
) -> dict:
    """Collects all runs for algorithms for a given benchmark."""
    assert path.exists(), f"{path} does not exist!"
    assert (path / f"benchmark={benchmark}").exists(), f"{path}/benchmark={benchmark} does not exist!"

    path = path / f"benchmark={benchmark}"

    all_algo_runs = {
        algo: collect_all_seeds(
            path / f"algorithm={algo}", seeds, continuations, wallclock, overhead, strict, analysis
        ) for algo in algorithms
    }
    return all_algo_runs


def collect(
        path: str | Path,
        benchmarks: List[str],
        algorithms: List[str],
        seeds: List[int] = None,
        continuations: bool = False,
        wallclock: bool = False, 
        overhead: bool = False, 
        strict: bool = False,
        analysis: bool = False,
) -> dict:
    """Collects all runs for specified arguments."""
    path = path if isinstance(path, Path) else Path(path)
    assert path.exists(), f"{path} does not exist!"

    all_runs = {
        bench: collect_algorithms(
            path, bench, algorithms, seeds, continuations, wallclock, overhead, strict, analysis
        ) for bench in benchmarks
    }
    return all_runs


def get_aggregated_plot_style(
        plot_data: dict,
        output_path: Path,
        filename: str,
        column_of_interest: str="inc_loss",
        log_x: bool = False,
        log_y: str = False,
        x_range: tuple = None,
        color_map: dict = None,
        marker_map: dict = None,
        marker_args: dict = None,
        label_map: dict = None, 
        wallclock: bool = False,
        overhead: bool = False,
        analysis: bool = False,

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
            # collecting the mean score across benchmarks
            algo_perf[algo][seed], _ = group_run_dataframes(
                list(seed_data.values()),
                column_of_interest=column_of_interest,
                analysis=analysis,
                x_range=x_range,
            )
        # averaging score across seeds
        algo_perf[algo]["mean"], algo_perf[algo]["sem"] = (
            group_run_dataframes(
                list(algo_perf[algo].values()),
                column_of_interest=column_of_interest,
                analysis=analysis,
                x_range=x_range,
            )
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
    plt.clf()

    algo_order = [(algo, algo_data["mean"].values[-1]) for algo, algo_data in algo_perf.items()]
    algo_order.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(1, 1, figsize=(PLOT_SIZE * 2, int(PLOT_SIZE * 1.5)))

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
    if log_x:
        ax.set_xscale("log")
    if x_range is not None:
        ax.set_xlim(*x_range)
    
    if wallclock:
        fig.supxlabel("Wallclock time (in s)")
    elif not wallclock and overhead: 
        fig.supxlabel("Only overhead time (in s)")
    else: 
        fig.supxlabel("Total epochs spent")
    ax.set_ylabel("Normalized regret") if not analysis else ax.set_ylabel(ANALYSIS_Y_LABEL[column_of_interest])

    # Move the legend to the right side of the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # fig.suptitle("Aggregate results across benchmarks")
    plt.tight_layout()

    target = output_path / f"{filename}.png"
    plt.savefig(target, bbox_inches='tight')
    print(f"\nPlot saved as {target}\n")


def get_per_benchmark_plot_style(
        plot_data: dict,
        output_path: Path,
        filename: str,
        log_x: bool = False,
        log_y: str = False,
        x_range: tuple = None,
        color_map: dict = None,
        marker_map: dict = None,
        marker_args: dict = None,
        label_map: dict = None,
        wallclock: bool = False,
        overhead: bool = False,
) -> None:
    """Plots multiple sub-plots for performance of each algorithm per benchmark."""
    # Calculating subplot dimensions dynamically
    num_benchmarks = len(plot_data.keys())
    for _n in reversed(range(1, MAX_N_COLS + 1)):
        if num_benchmarks % _n == 0:
            break
    n_cols = min(MAX_N_COLS, num_benchmarks)
    max_empty_slots = 1
    while True:
        n_rows = num_benchmarks // n_cols
        n_rows += int(num_benchmarks % n_cols != 0)
        if (n_rows * n_cols) - num_benchmarks > max_empty_slots:
            n_cols -= 1
        else:
            break
    print(f"\nRows: {n_rows}\nCols: {n_cols}\n")

    # Create a figure and a grid of subplots
    plt.clf()
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(PLOT_SIZE * n_cols, PLOT_SIZE * n_rows))

    # Remove extra subplots
    for i in range(num_benchmarks, n_rows * n_cols):
        fig.delaxes(axs.flatten()[i])

    # greate a colormap
    # cmap = CMAP
    # generate colors
    # colors = [cmap(i) for i in np.linspace(0, 1, len(plot_data[list(plot_data.keys())[0]]))]
    # if color_map is None or marker_map is None:
    #     l_colors, l_line_styles, l_markers = get_plot_styles(algo_perf)
    # else:
    l_colors = color_map
    l_markers = marker_map

    # # update index/fidelity based on requirement of wallclock time
    # if wallclock and overhead:
    #     aggregate_by = "wallclock_with_overhead"
    # elif wallclock:
    #     aggregate_by = "wallclock_without_overhead"

    # do the plotting
    bench_perf = dict()
    for i, (bench, bench_data) in enumerate(plot_data.items()):
        bench_perf[bench] = dict()
        ax = axs.flatten()[i] if n_rows * n_cols > 1 else axs
        for j, (algo, algo_data) in enumerate(bench_data.items()):
            bench_perf[bench][algo] = dict()
            _m, _s = group_run_dataframes(
                list(algo_data.values()), 
                x_range=x_range,
            )
            bench_perf[bench][algo]["mean"] = _m
            bench_perf[bench][algo]["sem"] = _s
            ax.step(
                bench_perf[bench][algo]["mean"].index.values,
                bench_perf[bench][algo]["mean"].values,
                color=l_colors[algo],
                marker=l_markers[algo],
                       ** marker_args,
                markevery=max(int(len(bench_perf[bench][algo]["mean"].index.values) / 15), 1),
                label=algo if not label_map and algo not in label_map else label_map[algo],
                where="post",
            )
            ax.fill_between(
                bench_perf[bench][algo]["mean"].index.values,
                bench_perf[bench][algo]["mean"].values - bench_perf[bench][algo]["sem"].values,
                bench_perf[bench][algo]["mean"].values + bench_perf[bench][algo]["sem"].values,
                facecolor=l_colors[algo],
                alpha=0.25,
                step="post",
            )
        ax.set_title(bench)
        ax.legend()

        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        if x_range is not None:
            ax.set_xlim(*x_range)
    # end of plotting loop

    fig.supylabel("Normalized regret")
    if wallclock:
        fig.supxlabel("Wallclock time (in s")
    elif not wallclock and overhead:
        fig.supxlabel("Only overhead time (in s)")
    else: 
        fig.supxlabel("Total epochs spent")
    plt.tight_layout()

    target = output_path / f"{filename}.png"
    plt.savefig(target)
    print(f"\nPlot saved as {target}\n")


def parse_args():
    parser = ArgumentParser(description="lc-pfn-hpo plotting")

    parser.add_argument("--basedir", type=str, default=None)
    parser.add_argument("--expgroup", type=str, default=None, required=True)

    parser.add_argument("--continuations", action="store_true")

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Expects a list of algorithms, comma separated str. Needs at least one."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Expects a list of benchmarks, comma separated str. Needs at least one."
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
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="If specified, returns one plot with the benchmarks aggregated into one."
    )
    parser.add_argument(
        "--rank",
        action="store_true",
        help="If specified, returns a plot with the rank of each algorithm."
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["baseline", "benchmark", "optimum", "none"],
        default="baseline",
        help="If specified, normalizes the regrets. Default is baseline."
    )
    parser.add_argument(
        "--plot_all",
        action="store_true",
        help=(
            "If specified, plots both per benchmark and aggregated runs. "
            "This flag overrides the --aggregate flag."
        )
    )

    parser.add_argument("--x_range", nargs=2, type=float, default=[1, 1000])
    parser.add_argument("--log_x", action="store_true")
    parser.add_argument("--log_y", action="store_true")
    parser.add_argument(
        "--y_range",
        nargs=2,
        type=float,
        default=None,
        help="Auto by default"
    )

    parser.add_argument("--filename", type=str, default=None, help="Name sans extension.")
    # TODO: make use of this argument
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If True, requires that no runs are missing."
    )

    # arguments for wallclock time
    parser.add_argument(
        "--wallclock",
        action="store_true",
        help="If True, plots x-axis as wallclock."
    )
    parser.add_argument(
        "--overhead",
        action="store_true",
        help="If True, along with wallclock=True, plots x-axis as wallclock + overhead."
    )

    # to toggle tueplots
    parser.add_argument(
        "--tueplots",
        action="store_true",
        help="If True, use tueplots styling for icml2024"
    )

    # to do analysis trace
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="If True, looks for analysis_data.csv for analysis traces"
    )

    return parser.parse_args()


import seaborn as sns

DEFAULT_MARKER_KWARGS = dict(
    markersize=10,
    fillstyle="full",
    markeredgewidth=3,
    markerfacecolor="white",
)
# Colorblind
COLORS = sns.color_palette("bright") + sns.color_palette("colorblind")
MARKERS = ['o', 'v', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '^', '<', '>']

color_map = {"random_search": COLORS[7],
             "pfn-bopfn-broken-unisep-pi-random": COLORS[0],
             "ifbo": COLORS[0],
             "pfn-bopfn-broken-expsep-pi-random": COLORS[9],
             "hyperband": COLORS[1],
             "asha": COLORS[5],
             "mf_ei_bo": COLORS[2],
             "dyhpo-neps-v2": COLORS[3],
             "dpl-neps-max": COLORS[4],
             "dyhpo-arlind": COLORS[3],
             "dpl-arlind": COLORS[4],
            # dpl
            "dpl-neps-pi": COLORS[4],
            "dpl-neps-pi-random": COLORS[6],
            "dpl-neps-pi-max": COLORS[8],
            "dpl-neps-ei": COLORS[10],
            "dpl-neps-ei-random-horizon": COLORS[12],
            "dpl-neps-ei-max": COLORS[14],
            "dpl-neps-ei-random": COLORS[16],
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
              "ifbo": MARKERS[1],
              "pfn-bopfn-broken-expsep-pi-random": MARKERS[1],
              "hyperband": MARKERS[2],
              "asha": MARKERS[3],
              "mf_ei_bo": MARKERS[4],
              "dyhpo-neps-v2": MARKERS[5],
              "dpl-neps-max": MARKERS[6],
              "dyhpo-arlind": MARKERS[5],
              "dpl-arlind": MARKERS[6],
            #   "pfn-bopfn-broken-unisep-pi-random": MARKERS[1],
            # dpl
            "dpl-neps-pi": MARKERS[4],
            "dpl-neps-pi-random": MARKERS[6],
            "dpl-neps-pi-max": MARKERS[8],
            "dpl-neps-ei": MARKERS[10],
            "dpl-neps-ei-random-horizon": MARKERS[5],
            "dpl-neps-ei-max": MARKERS[7],
            "dpl-neps-ei-random": MARKERS[9],
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

label_map = {"random_search": "Uniform BB",
             "pfn-bopfn-broken-unisep-pi-random": "ICL-FT-BO",
             "ifbo": "ifBO",
             # "pfn-bopfn-broken-expsep-pi-random": "CL-FT-BO",
             "hyperband": "Hyperband",
             "asha": "ASHA",
             "mf_ei_bo": "MF-BO",
             "dyhpo-neps-v2": "DyHPO",
             "dpl-neps-max": "DPL",
             "dyhpo-arlind": "DyHPO",
             "dpl-arlind": "DPL",
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
              # "pfn-bopfn-broken-expsep-pi-random",
              "hyperband",
              "asha",
              "mf_ei_bo"]
neps_v = ["dyhpo-neps-v2",
          "dpl-neps-max"]
arlind = ["dyhpo-arlind",
          "dpl-arlind"]

if __name__ == "__main__":
    args = parse_args()

    if args.overhead:
        raise ValueError("Overhead time is not yet supported.")

    if args.tueplots:
        from tueplots import bundles

        specs = bundles.icml2024()
        specs.update(TUEPLOTS_SPECS)

        plt.rcParams.update(specs)

        args.filename = f"tueplots_{args.filename}"

    # Basic argument checks
    if args.seeds is not None:
        assert len(args.seeds) <= 2, "Invalid --seeds. Check --help."
    assert len(args.algorithms) > 0, "Invalid --algorithms. Check --help."
    assert len(args.benchmarks) > 0, "Invalid --benchmarks. Check --help."
    assert args.x_range is not None and len(args.x_range) > 0, "Invalid --x_range. Check --help."
    if args.y_range is not None:
        assert len(args.y_range) > 0, "Invalid --y_range. Check --help."

    args.basedir = DEFAULT_BASE_PATH if args.basedir is None else Path(args.basedir)
    assert args.basedir.exists(), f"Base path: {args.basedir} does not exist!"

    target_path = args.basedir / args.expgroup
    assert target_path.exists(), f"Output target path: {target_path} does not exist!"

    output_path = args.basedir / ".." / "plots" / args.expgroup
    output_path.mkdir(parents=True, exist_ok=True)

    # Setting flags to determine plotting scope
    plot_per_benchmark = (not args.aggregate) | args.plot_all
    plot_aggregate = args.plot_all | args.aggregate
    plot_rank = args.rank | args.plot_all
    
    # Collecting (and preprocessing) data for plotting
    print("Collecting data for plotting...")
    plot_data = collect(
        target_path,
        args.benchmarks,
        args.algorithms,
        args.seeds,
        continuations=args.continuations,
        wallclock=args.wallclock,
        overhead=args.overhead,
        strict=args.strict,  # if True, will not plot if all seeds and runs are not present
        analysis=args.analysis,
    )

    # Check if analysis plots
    if args.analysis:
        print("Plotting analysis plots...")
        print("Mean...")
        get_aggregated_plot_style(
            plot_data.copy(),
            output_path,
            filename="min" if args.filename is None else f"min_{args.filename}",
            column_of_interest="mean_rank",
            log_x=args.log_x,
            log_y=args.log_y,
            x_range=args.x_range,
            color_map=color_map,
            marker_map=marker_map,
            marker_args=DEFAULT_MARKER_KWARGS,
            label_map=label_map,
            wallclock=args.wallclock,
            overhead=args.overhead,
            analysis=True,
        )
        print("Min...")
        get_aggregated_plot_style(
            plot_data.copy(),
            output_path,
            filename="mean" if args.filename is None else f"mean_{args.filename}",
            column_of_interest="min_rank",
            log_x=args.log_x,
            log_y=args.log_y,
            x_range=args.x_range,
            color_map=color_map,
            marker_map=marker_map,
            marker_args=DEFAULT_MARKER_KWARGS,
            label_map=label_map,
            wallclock=args.wallclock,
            overhead=args.overhead,
            analysis=True,
        )
        print("Intersection over double set...")
        get_aggregated_plot_style(
            plot_data.copy(),
            output_path,
            filename="idouble" if args.filename is None else f"idouble_{args.filename}",
            column_of_interest="intersection_double",
            log_x=args.log_x,
            log_y=args.log_y,
            x_range=args.x_range,
            color_map=color_map,
            marker_map=marker_map,
            marker_args=DEFAULT_MARKER_KWARGS,
            label_map=label_map,
            wallclock=args.wallclock,
            overhead=args.overhead,
            analysis=True,
        )
        print("Intersection over top 99 percentile...")
        get_aggregated_plot_style(
            plot_data.copy(),
            output_path,
            filename="i99" if args.filename is None else f"i99_{args.filename}",
            column_of_interest="intersection_p99",
            log_x=args.log_x,
            log_y=args.log_y,
            x_range=args.x_range,
            color_map=color_map,
            marker_map=marker_map,
            marker_args=DEFAULT_MARKER_KWARGS,
            label_map=label_map,
            wallclock=args.wallclock,
            overhead=args.overhead,
            analysis=True,
        )
        print("Exitting...")
        sys.exit(1)

    # Plotting per-benchmark
    if plot_per_benchmark:
        print("Plotting per benchmark plots...")
        # reorder data
        get_per_benchmark_plot_style(
            plot_data.copy(),
            output_path,
            filename="per-bench" if args.filename is None else f"per-bench_{args.filename}",
            log_x=args.log_x,
            log_y=args.log_y,
            x_range=args.x_range,
            color_map=color_map,
            marker_map=marker_map,
            marker_args=dict(
                markersize=6,
                fillstyle="full",
                # markeredgewidth=3,
                markerfacecolor="white",
            ),
            label_map=label_map,
            wallclock=args.wallclock,
            overhead=args.overhead,
        )

    # Normalizing incumbents
    name_normalization = ""
    if args.normalize in ["baseline", "benchmark", "optimum"]:
        if args.normalize == "baseline":
            bounds = calc_bounds_per_benchmark(plot_data)
            name_normalization = "normBaseline"
        elif args.normalize == "benchmark":
            benchmark_name = args.benchmarks[0].split("-")[0]
            with open(args.basedir / ".." / f"benchmarks_bounds/{benchmark_name}.json", "r") as f:
                bounds = json.load(f)
            name_normalization = "normBenchmark"
        plot_data = normalize_and_calculate_regrets(plot_data, bounds)
    elif args.normalize == "none":
        name_normalization = "noNorm"
    else:
        raise ValueError(f"Invalid normalization: {args.normalize}")

    # Plotting aggregated plots
    if plot_aggregate:
        print("Plotting aggregated plot...")
        # reorder data
        get_aggregated_plot_style(
            plot_data.copy(),
            output_path,
            filename="aggregated" if args.filename is None else f"aggregated_{args.filename}_{name_normalization}",
            log_x=args.log_x,
            log_y=args.log_y,
            x_range=args.x_range,
            color_map=color_map,
            marker_map=marker_map,
            marker_args=DEFAULT_MARKER_KWARGS,
            label_map=label_map,
            wallclock=args.wallclock,
            overhead=args.overhead,
        )
    if plot_rank:
        print("Plotting rank plot...")
        get_rank_plot(
            plot_data.copy(),
            output_path,
            filename="rank" if args.filename is None else f"rank_{args.filename}",
            x_range=args.x_range,
            color_map=color_map,
            marker_map=marker_map,
            marker_args=DEFAULT_MARKER_KWARGS,
            label_map=label_map,
            wallclock=args.wallclock,
            overhead=args.overhead,
        )

# end of main
