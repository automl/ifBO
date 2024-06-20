from __future__ import annotations

from copy import deepcopy
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
from typing_extensions import List

from tueplots import bundles, figsizes, fontsizes
plt.rcParams.update(bundles.icml2024(column="full"))


MAX_N_COLS = 3
PLOT_SIZE = 5
CMAP = plt.get_cmap('tab10')
LINESTYLES = ['-', '--', '-.', ':']
MARKERS = ['o', '<', 's', '*', 'H', '+', 'D', 'd', '|', '_']


def smooth_minpool(data: list, window_size: int=3) -> list:
    smoothed_data = []
    for i in range(len(data)):
        # Define window boundaries
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        # Get the window of data
        window = data[start:end]
        # Take the maximum value within the window
        smoothed_data.append(min(window))
    return smoothed_data


def smooth_gaussian(data: list, sigma: int=1) -> list:
    smoothed_data = gaussian_filter1d(data, sigma)
    return smoothed_data.tolist()


def group_run_dataframes(
    df_list: List[pd.DataFrame],
    column_of_interest: str="inc_loss",
    aggregate_by: str="cumsum_fidelity",
    **kwargs
):
    """Given a list of dataframes, collates them based on the index."""
    assert len(df_list), "Empty list! Needs at least one element as a pd.DataFrame!"
    list_status = all(isinstance(element, (pd.DataFrame, pd.Series)) for element in df_list)
    assert list_status, "All elements in the list are not pandas data types!"

    # sets the index to be the cumulative fidelities and filters for the chosen columns
    if isinstance(df_list[0], pd.DataFrame):
        df_list = [
            _df.set_index(aggregate_by).loc[:, column_of_interest] for _df in df_list
        ]
        # if a list of pd.Series, likely a grouping was performed already
        # TODO: more strict check? check all the _df have the same length.
        # assert all(df_list[0].shape[0] == _df.shape[0] for _df in df_list[1:])

    # filter for the range of interest
    x_range = kwargs.get("x_range", None)
    if x_range is not None:
        df_list = [df.loc[x_range[0]:x_range[1]] for df in df_list]
    if sum([any(_df.index.values > x_range[1]) for _df in df_list]):
        raise ValueError("Some dataframes have indices greater than the specified range!")
    # get the last/max index
    max_index = max(set().union(*[set(df.index) for df in df_list]))
    # retains only the rows where incumbent values are updated, helps save memory
    df_list = [df[df.diff().ne(0)] for df in df_list]

    # remove spurious NaNs from index
    df_list = [df.loc[df.index.dropna()] for df in df_list]

    # removes duplicate indices
    df_list = [df.loc[~df.index.duplicated(keep="first")] for df in df_list]

    # collects the unique values seen for the x-axis
    union_index = pd.Index(set([max_index]).union(*[set(df.index) for df in df_list])).sort_values()

    # equalizes all data frames to have this unique list of x-axis values accounted for
    df_values = np.array([
        df.reindex(union_index, method='ffill').sort_index().values for df in df_list
    ])

    # smooth for analysis
    if kwargs.get("analysis", False):
        df_values = np.array(smooth_gaussian(df_values, sigma=0.35))

    if kwargs.get("nanmean", False):
        df_values_ = np.nan_to_num(df_values, nan=1)
        mean_df = pd.Series(df_values_.mean(axis=0), index=union_index).sort_index() 
        sem_df = pd.Series(sem(df_values_, axis=0), index=union_index).sort_index()
    else:
        mean_df = pd.Series(df_values.mean(axis=0), index=union_index).sort_index()
        sem_df = pd.Series(sem(df_values, axis=0), index=union_index).sort_index()

    return mean_df, sem_df


def reorder_for_per_benchmark_ranking_plots(plot_data: dict) -> dict:
    """For each see benchmark reorders to have seed and then algorithms."""
    # this ordering should allow us to calculate relative ranks across a set of algorithms
    # on a benchmark for a seed, following which we can retrieve the variation of these
    # ranks across seed for each benchmark
    reordered_data = {}
    for benchmark, algo_dict in plot_data.items():
        reordered_data[benchmark] = {}
        for algo, seed_dict in algo_dict.items():
            for seed, df in seed_dict.items():
                seed_data = reordered_data[benchmark].setdefault(seed, {})
                seed_data[algo] = df
    return reordered_data


def reorder_for_aggregated_ranking_plots(plot_data: dict) -> dict:
    """For each see seed reorders to have benchmarks and then algorithms."""
    # this ordering should allow us to calculate relative ranks across a set of algorithms
    # on a benchmark for a seed, following which we can average across benchmarks for the
    # same set of algorithms and then finally take the variation across seeds
    reordered_data = {}
    for benchmark, algo_dict in plot_data.items():
        reordered_data[benchmark] = {}
        for algo, seed_dict in algo_dict.items():
            for seed, df in seed_dict.items():
                seed_data = reordered_data[benchmark].setdefault(seed, {})
                seed_data[algo] = df
    return reordered_data


def reorder_for_aggregated_benchmark_plots(plot_data: dict) -> dict:
    """Reorders such that for each algo, there are seeds and then benchmarks."""
    reordered_data = {}
    for benchmark, algo_dict in plot_data.items():
        for algo, seed_dict in algo_dict.items():
            for seed, df in seed_dict.items():
                algo_data = reordered_data.setdefault(algo, {})
                seed_data = algo_data.setdefault(seed, {})
                # algo_data = seed_data.setdefault(algo, {})
                seed_data[benchmark] = df
    return reordered_data


def calc_bounds_per_benchmark(data: dict) -> dict:
    """Calculate min-max bounds per benchmark."""

    def func_to_parallelize(seed_data):
        # returns the lowest,highest loss seen
        l_bound = seed_data["inc_loss"].values[-1]
        u_bound = seed_data["inc_loss"].values[0]
        return l_bound, u_bound

    bounds = dict()
    for benchmark, bench_data in data.items():
        bounds[benchmark] = [np.inf, -np.inf]  # (min_bound, max_bound)
        for _, algo_data in bench_data.items():
            # NOTE: will use ALL available CPUs
            collected_values = np.array(Parallel(n_jobs=-1)(
                delayed(func_to_parallelize)
                (seed_data) for _, seed_data in algo_data.items()
            ))
            bounds[benchmark][0] = min(bounds[benchmark][0], collected_values.min())
            bounds[benchmark][1] = max(bounds[benchmark][1], collected_values.max())
    print(f"Bounds:\n{bounds}\n")
    return bounds


def normalize_and_calculate_regrets(data: dict, bounds: dict) -> dict:
    """Normalizes the incumbent trace based on min-max bounds per benchmark."""
    missing_set = set(data.keys()) - set(bounds.keys())
    assert not len(missing_set), f"No bounds found for {missing_set}!"

    def func_to_parallelize(seed, seed_data, l_bound, u_bound):
        # returns the data frame with the incumbent trace normalized
        seed_data.loc[:, "inc_loss"] = (seed_data["inc_loss"] - l_bound) / (u_bound - l_bound)
        return seed, seed_data

    for benchmark, bench_data in data.items():
        algo_map = dict()
        for algo, algo_data in bench_data.items():
            # NOTE: will use ALL available CPUs
            algo_map[algo] = dict(Parallel(n_jobs=-1)(
                delayed(func_to_parallelize)
                (seed, seed_data, *bounds[benchmark]) for seed, seed_data in algo_data.items()
            ))
        data[benchmark] = deepcopy(algo_map)

    return data


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_plot_styles(algo_perf: dict):
    # Define a list of line styles and markers
    line_styles = LINESTYLES
    markers = MARKERS

    # Get the colormap
    cmap = sns.hls_palette(len(algo_perf.keys()))

    # Create a dictionary to store color, line style, and marker for each prefix
    prefix_algo, prefix_acq = {}, {}
    l_colors, l_line_styles, l_markers = {}, {}, {}

    id_algo, id_acq = 0, 0

    for algo in algo_perf.keys():
        name = algo.split("-")
        algo_name = ""
        acq_name = ""
        if len(name) == 1: 
            algo_name = name[0]
        elif len(name) == 2:
            algo_name = name[0] + "-" + name[1]
        elif len(name) >= 3:
            algo_name = "-".join(name[:-1])
            acq_name = name[-1]
        else:
            raise ValueError("Invalid algorithm name")

        if algo_name not in prefix_algo:
            prefix_algo[algo_name] = cmap[id_algo]
            id_algo += 1
        
        if acq_name not in prefix_acq:
            prefix_acq[acq_name] = {
                "line_style": line_styles[id_acq % len(line_styles)],
                "marker": markers[id_acq % len(markers)]
            }
            id_acq += 1
        
        l_colors[algo] = prefix_algo[algo_name]
        l_line_styles[algo] = prefix_acq[acq_name]["line_style"]
        l_markers[algo] = prefix_acq[acq_name]["marker"]
    
    return l_colors, l_line_styles, l_markers


def get_aggregated_plot(
    plot_data: dict,
    output_path: Path,
    filename: str,
    log_x: bool=False,
    log_y: str=False,
    x_range: tuple=None,
    wallclock: bool=False,
    overhead: bool=False,
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
                x_range=x_range,
            )
        # averaging score across seeds
        algo_perf[algo]["mean"], algo_perf[algo]["sem"] = (
            group_run_dataframes(
                list(algo_perf[algo].values()),
                x_range=x_range,
            )
        )
        # removing seed keys
        _ = [algo_perf[algo].pop(_seed) for _seed in seeds]

    # Define a list of line styles and markers
    l_colors, l_line_styles, l_markers = get_plot_styles(algo_perf)

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
            markersize=6,
            markevery=max(1, int(len(algo_data["mean"].index.values) / 15)),
            fillstyle='none',
            label=algo,
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
    ax.set_ylabel("Normalized regret")

    # Move the legend to the right side of the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle("Aggregate results across benchmarks")
    plt.tight_layout()

    target = output_path / f"{filename}.png"
    plt.savefig(target, bbox_inches='tight')
    print(f"\nPlot saved as {target}\n")


def get_per_benchmark_plot(
    plot_data: dict,
    output_path: Path,
    filename: str,
    log_x: bool=False,
    log_y: str=False,
    x_range: tuple=None,
    wallclock: bool=False,
    overhead: bool=False,
) -> None:
    """Plots multiple sub-plots for performance of each algorithm per benchmark."""
    # Calculating subplot dimensions dynamically
    num_benchmarks = len(plot_data.keys())
    for _n in reversed(range(1, MAX_N_COLS+1)):
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
    cmap = CMAP
    # generate colors
    colors = [cmap(i) for i in np.linspace(0, 1, len(plot_data[list(plot_data.keys())[0]]))]
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
                color=colors[j],
                label=algo,
                where="post",
            )
            ax.fill_between(
                bench_perf[bench][algo]["mean"].index.values,
                bench_perf[bench][algo]["mean"].values - bench_perf[bench][algo]["sem"].values,
                bench_perf[bench][algo]["mean"].values + bench_perf[bench][algo]["sem"].values,
                facecolor=colors[j],
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

    if wallclock:
        fig.supxlabel("Wallclock time (in s)")
    elif not wallclock and overhead: 
        fig.supxlabel("Only overhead time (in s)")
    else: 
        fig.supxlabel("Total epochs spent")
    ax.set_ylabel("Normalized regret")

    plt.tight_layout()
    
    target = output_path / f"{filename}.png"
    plt.savefig(target)
    print(f"\nPlot saved as {target}\n")


def get_rank_plot(
    plot_data: dict,
    output_path: Path,
    filename: str,
    x_range: tuple=None,
    color_map: dict = None,
    marker_map: dict = None,
    marker_args: dict = None,
    label_map: dict = None,
    wallclock: bool = False,
    overhead: bool = False,
) -> None:

    plt.rcParams.update(figsizes.icml2024_full(nrows=1, ncols=1, height_to_width_ratio=.8))
    plt.rcParams.update(fontsizes.icml2024())
    plt.rcParams['xtick.labelsize'] = 100  # Adjust the size to your preference
    plt.rcParams['ytick.labelsize'] = 100  # Adjust the size to your preference
    plt.rcParams['axes.labelsize'] = 100

    # fig, ax = plt.subplots(1, 1 , figsize=(PLOT_SIZE * 2, int(PLOT_SIZE * 1.5)))
    fig, ax = plt.subplots(1, 1 , figsize=(10 * 2, int(10 * 1.5)))   
    df_rank = pd.DataFrame()

    l_datasets = list(plot_data.keys())
    l_algos = plot_data[l_datasets[0]].keys()
    l_colors, l_line_styles, l_markers = get_plot_styles({algo: [] for algo in l_algos})

    aggregate_by = "cumsum_fidelity"
    # if wallclock and overhead:
    #     aggregate_by = "wallclock_with_overhead"
    # elif wallclock and not overhead:
    #     aggregate_by = "wallclock_without_overhead"
    # elif not wallclock and overhead:
    #     aggregate_by = "overhead"

    for dataset in plot_data.keys():

        ldf = []

        for algo in l_algos:
            for seed in range(len(plot_data[dataset][algo])):
                df = plot_data[dataset][algo][seed][[aggregate_by, "inc_loss"]].set_index(aggregate_by, drop=True)
                df = df.loc[df.index.dropna()]  # removing NaNs in index
                df = df.loc[~df.index.duplicated(keep="first")]  # removing duplicate index
                df = df.reindex(np.arange(1, x_range[1] + 1), method="ffill").sort_index()
                # df = df.reindex(np.arange(1, x_range[1]), method="ffill").fillna(1).reset_index()
                df["model"] = algo
                df["seed"] = seed
                ldf.append(df)

        # df = pd.concat(ldf, ignore_index=True)
        df = pd.concat(ldf, ignore_index=False).reset_index()
        df = df.pivot(columns="model", index=[aggregate_by, "seed"], values="inc_loss").rank(ascending=True, axis=1)
        df = df.melt(value_vars=l_algos, value_name="rank", ignore_index=False).reset_index()
        df["dataset"] = dataset
        df_rank = pd.concat([df_rank, df], axis=0, ignore_index=True)
    
    df_rank = df_rank.groupby([aggregate_by, "model"])["rank"].agg(['mean','sem'])
    df_rank = df_rank.reset_index().sort_values(aggregate_by, ascending=True)
    ordered_algos = df_rank[df_rank.loc[:, aggregate_by] == df_rank.loc[: ,aggregate_by].max()][["model", "mean"]].sort_values("mean", ascending=False).model.tolist()

    new_indices = np.linspace(start=0, stop=1000000, endpoint=True, num=100) 
    _mean_rank = np.mean(np.arange(1, len(ordered_algos)+1))  # .astype(int)
    _dummy_row = pd.Series([_mean_rank], index=[0])

    import os
    _files = os.listdir()
    if "lcbench.parquet.gzip" not in _files and "pd1.parquet.gzip" not in _files and "taskset.parquet.gzip" not in _files and "all.parquet.gzip" not in _files:
        df_rank.to_parquet("./lcbench.parquet.gzip")
        print("Saving ./lcbench.parquet.gzip")    
    elif "lcbench.parquet.gzip" in _files and "pd1.parquet.gzip" not in _files and "taskset.parquet.gzip" not in _files and "all.parquet.gzip" not in _files:    
        df_rank.to_parquet("./pd1.parquet.gzip")
        print("Saving ./pd1.parquet.gzip")    
    elif "lcbench.parquet.gzip" in _files and "pd1.parquet.gzip" in _files and "taskset.parquet.gzip" not in _files and "all.parquet.gzip" not in _files:
        df_rank.to_parquet("./taskset.parquet.gzip")
        print("Saving ./taskset.parquet.gzip")    
    elif "lcbench.parquet.gzip" in _files and "pd1.parquet.gzip" in _files and "taskset.parquet.gzip" in _files and "all.parquet.gzip" not in _files:
        df_rank.to_parquet("./all.parquet.gzip")
        print("Saving ./all.parquet.gzip")    


    for i, algo in enumerate(ordered_algos):
        _new_indices = new_indices[new_indices <= df_rank[df_rank.model == algo]["mean"].index.max()]  # keeping only the times see

        _df_mean = pd.concat((_dummy_row, df_rank[df_rank.model == algo]["mean"])).sort_index()
        _df_mean = _df_mean.loc[~_df_mean.index.duplicated(keep="first")]
        _df_sem = pd.concat((_dummy_row, df_rank[df_rank.model == algo]["sem"])).sort_index()
        _df_sem = _df_sem.loc[~_df_sem.index.duplicated(keep="first")]

        _mean = _df_mean.reindex(_new_indices, method="ffill") 
        _sem = _df_sem.reindex(_new_indices, method="ffill") 

        ax.plot(   
            # df_rank[df_rank.model == algo].loc[:, aggregate_by].values, 
            _mean.index.values,
            _mean.values, 
            # df_rank[df_rank.model == algo]["mean"].values,
            color=color_map[algo],
            marker=marker_map[algo],
            markevery=50,
            **marker_args,
            label=label_map[algo],
            linewidth=20,
        )
        ax.fill_between(
            # df_rank[df_rank.model == algo].loc[:, aggregate_by].values,
            _mean.index.values,
            _mean.values - _sem.values,
            _mean.values + _sem.values,
            # df_rank[df_rank.model == algo]["mean"].values - df_rank[df_rank.model == algo]["sem"].values, 
            # df_rank[df_rank.model == algo]["mean"].values + df_rank[df_rank.model == algo]["sem"].values,
            facecolor=color_map[algo],
            alpha=0.1,
            step="post"
        )
    ax.set_ylim(1, len(l_algos))
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', labelsize=100)
    ax.tick_params(axis='y', labelsize=100)

    fig.legend(prop={"size": 100})

    fig.supylabel("Rank", fontsize=100)
    
    if wallclock:
        fig.supxlabel("Wallclock time (in s)", fontsize=100)
    elif not wallclock and overhead: 
        fig.supxlabel("Only overhead time (in s)", fontsize=100)
    else: 
        fig.supxlabel("Total epochs spent", fontsize=100)
    
    # if wallclock or overhead:
    #     ax.set_xscale("log")

    plt.tight_layout()
    
    target = output_path / f"{filename}.png"
    plt.savefig(target)
    print(f"\nPlot saved as {target}\n")

