import argparse
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from pfns_hpo.regret_plot import collect, parse_args


label_map = {"random_search": "Uniform BB",
             "pfn-bopfn-broken-unisep-pi-random": "ICL-FT-BO",
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

HERE = Path(__file__).parent.absolute()
DEFAULT_BASE_PATH = HERE.parent / "results"


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


def _sanity_check(df: pd.DataFrame) -> None:
    assert all(df.Status == "Complete"), "Not all configs are complete!"


def get_best_config_id(df: pd.DataFrame) -> str:
    best_config_id = int(df.loc[df["result.loss"].argmin()]["Config_id"].split("_")[0])
    return best_config_id


def get_num_unique_samples(df: pd.DataFrame) -> int:
    val = df.Config_id.apply(lambda x: int(x.split("_")[0])).max()
    return val


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=['metadata.time_sampled', 'metadata.time_end'])
    return df


def get_num_inc_updates(df: pd.DataFrame) -> int:
    df = sort_by_time(df)
    inc_trace = pd.Series(np.minimum.accumulate(df["result.loss"].values))
    num_updates = (inc_trace.diff() != 0).sum()
    return num_updates


def get_num_switches(df: pd.DataFrame) -> int:
    df = sort_by_time(df)
    df["only_id"] = df.Config_id.apply(lambda x: int(x.split("_")[0])).values
    num_switches = sum(df.only_id.diff() != 0)
    return num_switches


def _get_highest_fidelity_seen(df: pd.DataFrame, id: int) -> int:
    assert isinstance(df.index[0], tuple), "Index should be multi-level!"
    return df.loc[id].index.get_level_values(0).sort_values()[-1]


def _get_fidelity_with_best_score(df: pd.DataFrame, id: int) -> int:
    assert isinstance(df.index[0], tuple), "Index should be multi-level!"
    _z_max = _get_highest_fidelity_seen(df, id)
    _lc = df.loc[(id, _z_max)]["result.info_dict.learning_curves.valid"]


def reorder_data_multi_index(df: pd.DataFrame) -> pd.DataFrame:
    df = sort_by_time(df)
    df["id"] = df.Config_id.apply(lambda x: int(x.split("_")[0])).values
    df["fid"] = df.Config_id.apply(lambda x: int(x.split("_")[1])).values
    df = df.set_index(["id", "fid"]).sort_index()
    lc_key = "result.info_dict.learning_curves.valid"
    df[lc_key] = df[lc_key].apply(eval)
    return df


def get_steps_spent_after_best(df: pd.DataFrame, smooth: str=None) -> dict:
    df = reorder_data_multi_index(df)
    _data = dict()
    for id in df.index.get_level_values(0).unique():
        _z_max = _get_highest_fidelity_seen(df, id)
        _lc = df.loc[(id, _z_max)]["result.info_dict.learning_curves.valid"]
        if smooth is not None:
            if smooth == "gaussian":
                _smooth_lc = smooth_gaussian
            elif smooth == "minpool":
                _smooth_lc = smooth_minpool
            else:
                raise ValueError("Invalid smoothing method!")
            _lc = _smooth_lc(_lc)
        _data[id] = _z_max - np.argmin(_lc)
    return _data


def get_budget_spent_after_best(df: pd.DataFrame, smooth: str=None) -> dict:
    data = get_steps_spent_after_best(df, smooth)
    return sum(data.values()) / len(df)


def get_thawing_age_trace(df: pd.DataFrame) -> list:
    df = sort_by_time(df)
    df["id"] = df.Config_id.apply(lambda x: int(x.split("_")[0])).values
    df["thaw_age"] = np.arange(1, len(df)+1) - df.id.values
    df["thaw_age_max"] = np.maximum.accumulate(df["thaw_age"])
    return df.thaw_age_max.values.tolist()


def _find_intervals(series: pd.Series, num: int) -> list:
    indices = series[series == num].index
    diffs = indices.to_series().diff()
    return diffs[diffs > 1].dropna().tolist()


def reselection_frequency(df: pd.DataFrame) -> pd.Series:
    df = sort_by_time(df)
    df["id"] = df.Config_id.apply(lambda x: int(x.split("_")[0])).values
    df["fid"] = df.Config_id.apply(lambda x: int(x.split("_")[1])).values

    all_intervals = Parallel(n_jobs=2)(
        delayed(_find_intervals)(df.id, _id)
        for _id in df.id.unique()
    )
    all_intervals = [_elem for elem in all_intervals for _elem in elem]
    return pd.Series(all_intervals)


def _compute_non_point_estimates(alg: str, plot_data: dict):
    return alg, pd.concat(
        Parallel(n_jobs=-1)(
            delayed(reselection_frequency)(plot_data[bench][alg][seed])
            for bench in plot_data if alg in plot_data[bench] for seed in plot_data[bench][alg]
        ), ignore_index=True
    )


def _get_point_estimates(df: pd.DataFrame, seed: int=None) -> pd.DataFrame:
    _sanity_check(df)
    point_estimates = dict()
    point_estimates["unique_samples"] = get_num_unique_samples(df)
    point_estimates["best_config_id"] = get_best_config_id(df)
    point_estimates["inc_updates"] = get_num_inc_updates(df) / len(df)
    point_estimates["num_switches"] = get_num_switches(df)
    point_estimates["div_raw"] = get_budget_spent_after_best(df)
    point_estimates["div_pool"] = get_budget_spent_after_best(df, "minpool")
    point_estimates["div_gauss"] = get_budget_spent_after_best(df, "gaussian")
    return pd.DataFrame(point_estimates, index=[seed])


if __name__ == "__main__":    
    args = parse_args()

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
    
    # Collecting (and preprocessing) data for plotting
    print("Collecting data for plotting...")
    plot_data = collect(
        target_path,
        args.benchmarks,
        args.algorithms,
        args.seeds,
        continuations=args.continuations,
        strict=args.strict  # if True, will not plot if all seeds and runs are not present
    )
    # plot_data = collect(
    #     "/work/dlclarge1/mallik-lcpfn-hpo/pfns_hpo/results/valid_bench",  # TODO: Change this path
    #     ["lcbench-tabular-Fashion-balanced", "lcbench-tabular-blood-balanced", "pd1-tabular-imagenet_resnet_512-5", "taskset-tabular-normalized-nlp-3-4p"],
    #     ["dpl-neps-max", "dyhpo-neps-v2", "pfn-bopfn-broken-unisep-pi-random"],
    #     [10],
    #     continuations=True,
    #     strict=True
    # )

    # calculate all non-trace point estimates per run
    point_estimates = dict()
    for bench in plot_data.keys():
        point_estimates[bench] = dict()
        for algo in plot_data[bench].keys():

            all_seeds = Parallel(n_jobs=-1)(
                delayed(_get_point_estimates)(seed_df, seed)
                for seed, seed_df in plot_data[bench][algo].items()
            )
            all_seeds = pd.concat(all_seeds)
            point_estimates[bench][algo] = all_seeds
    
    # concatenate across all benchmarks and seeds, per algo: # benchmark x # seeds
    point_estimates = {
        alg: pd.concat(
            [point_estimates[bench][alg] for bench in point_estimates if alg in point_estimates[bench]],
            ignore_index=True
        ) for alg in set(algo for bench in point_estimates for algo in point_estimates[bench])
    }
    point_estimates = {alg: _data.to_dict(orient="list") for alg, _data in point_estimates.items()}
    
    # calculate all non-trace point estimates per run
    non_point_estimates = dict(
        Parallel(n_jobs=-1)(
            delayed(_compute_non_point_estimates)(alg, plot_data)
            for alg in set(algo for bench in plot_data for algo in plot_data[bench])
        )
    )
    # adding resel_freq to point estimates
    for alg, series in non_point_estimates.items():
        point_estimates[alg].update({"resel_freq": series.values.tolist()})

    # plot 'em all
    title_map = {
        'unique_samples': "Number of unique samples collected",
        'best_config_id': "Sample ID of the incumbent found",
        'inc_updates': "Number of times the incumbent score was updated",
        'num_switches': "Number of times a different hyperparameter was selected",
        'resel_freq': "Unit of budget spent before reselecting a sampled hyperparameter",
        'div_raw': "% budget spent on a hyperparameter after best value observed (raw)",
        'div_pool': "% budget spent on a hyperparameter after best value observed (minpool)",
        'div_gauss': "% budget spent on a hyperparameter after best value observed (gaussian)",
    }
    metrics = list(point_estimates[list(point_estimates.keys())[0]].keys())
    for metric in metrics:
        print("Plotting metric:", metric)
        plt.figure()
        val = []
        ticklabel = []
        algo_list = []
        for algo, df in point_estimates.items():
            algo_list.append(algo)
            val.append(df[metric])
            ticklabel.append(label_map[algo])
        bp = plt.boxplot(val)

        if len([_alg for _alg in algo_list if "pfn" in _alg or "icl" in _alg.lower()]) == 1:
            # Find our algo index
            idx = ticklabel.index(label_map[[
                _alg for _alg in algo_list 
                if "pfn" in _alg.lower() or "icl" in _alg.lower()
            ][0]])
            # Calculate the mean and max tick of the first boxplot
            upper_whisker = bp["whiskers"][2*idx + 1].get_ydata()[1]
            median = bp["medians"][idx].get_ydata()[1]
            # Draw horizontal lines at the mean and upper_whisker
            plt.axhline(median, color='g', linestyle='--', linewidth=1.5)
            plt.axhline(upper_whisker, color='g', linestyle='dotted', linewidth=1.5)

        plt.xticks(np.arange(1, len(val)+1), ticklabel, rotation=5)
        plt.title(title_map[metric])
        if metric == "resel_freq":
            plt.yscale("log")
        plt.savefig(output_path / f"{args.filename}_{metric}.png")  # TODO: Change this path
