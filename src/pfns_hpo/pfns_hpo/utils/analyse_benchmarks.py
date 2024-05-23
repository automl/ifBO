from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

import mfpbench


BASE_PATH = Path(__file__).resolve().parent / ".." / "configs" / "benchmark"
BENCH_DATA_PATH = Path(__file__).resolve().parent / ".." / ".." / "data"
SEED = 12345


def percentage_budget_maxima(table: pd.DataFrame, value_metric: str) -> dict:
    data = dict()
    stats = dict()
    for idx in table.index.get_level_values(0).unique():
        data[idx] = table.loc[idx, :].loc[:, value_metric]
        if "error" in value_metric or "loss" in value_metric:
            data[idx] = 1 - data[idx]
        try:
            data[idx] = data[idx].argmax()
        except:
            breakpoint()
    data = pd.Series(data)
    # fraction of max fidelity
    data = data / table.index.get_level_values(1).max()
    stats.update(dict(
        low=data.where(data <= 0.33).dropna().shape[0] / data.shape[0],
        mid=data.where((data > 0.33) & (data < 0.66)).dropna().shape[0] / data.shape[0],
        high=data.where(data >= 0.66).dropna().shape[0] / data.shape[0],
    ))
    return stats


def is_monotonic(val: pd.Series) -> bool:
    _bool_val = all(val.diff().fillna(0) >= 0)
    return _bool_val


def percentage_monotonic(
    table: pd.DataFrame, value_metric: str, plot_mono: bool = False, name: str = None
) -> float:
    data = dict()
    for idx in table.index.get_level_values(0).unique():
        data[idx] = is_monotonic(table.loc[idx, :].loc[:, value_metric])
    data = pd.Series(data)
    per_mono = data.where(data == True).dropna().shape[0] / data.shape[0]
    if plot_mono:
        plot_dir = BASE_PATH / "analysis_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.clf()
        _df = table.loc[data.where(data == True).dropna().index]
        for idx in _df.index.get_level_values(0).unique():
            lc = _df.loc[idx, :].loc[:, value_metric].copy()
            # lc = np.clip(lc.fillna(np.inf), 0, 1)
            plt.plot(lc, color="black", alpha=0.5)
        plt.savefig(plot_dir / f"{name}-mono.png")
    return per_mono


def margin_of_best_curve(
    table: pd.DataFrame, value_metric: str, plot_margin: bool = False, name: str = None
) -> float:
    # adjust metric
    if "error" in value_metric or "loss" in value_metric:
        table.loc[:, value_metric] = 1 - table.loc[:, value_metric]
    # find best
    best_idx, best_step = table.index[table.loc[:, value_metric].argmax()]
    val = table.loc[best_idx, :].loc[:best_step, value_metric]
    val_full = table.loc[best_idx, :].loc[:, value_metric]
    val_diff = val.diff().fillna(0).abs()
    _weights = val.index.values
    # weighted mean
    err_margin = sum(val_diff.values * _weights) / sum(_weights)
    upper_bound = val.max()
    lower_bound = val.max() - err_margin
    data = dict()
    for idx in table.index.get_level_values(0).unique():
        _val_full = table.loc[idx, :].loc[:, value_metric]
        if _val_full.max() >= lower_bound:
            data[idx] = val_full.values
    data = pd.Series(data)

    fraction = data.shape[0] / table.shape[0]

    if plot_margin:
        plot_dir = BASE_PATH / "analysis_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.plot(val_full, label="orig", color="black")
        for idx in data.index.get_level_values(0).unique():
            plt.plot(data.loc[idx], color="black", alpha=0.5)
        plt.fill_between(
            np.arange(len(val_full.values)+1),
            upper_bound,
            lower_bound,
            color="red",
            alpha=0.25
        )
        plt.savefig(plot_dir / f"{name}-margin.png")

    return fraction


def plot_all_lcs(table: pd.DataFrame, value_metric: str, bench_name: str) -> None:
    plt.clf()
    for idx in table.index.get_level_values(0).unique():
        lc = table.loc[idx, value_metric].values
        # lc = np.clip(lc.fillna(np.inf), 0, 1)
        if "error" in value_metric or "loss" in value_metric:
            lc = 1 - lc
        plt.plot(lc, color="black", alpha=0.2)
    plot_dir = BASE_PATH / "analysis_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"{bench_name}-all.png")
    return


if __name__ == "__main__":
    print(BASE_PATH)
    print(BENCH_DATA_PATH)

    all_data = dict()
    for path in BASE_PATH.iterdir():
        if not str(path).endswith(".yaml") or not "tabular" in str(path.name):
            continue
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        bench_family = '-'.join(str(path.name).split('-')[:2])
        bench_name = str(path.name).split('.')[0]
        data["api"]["datadir"] = BENCH_DATA_PATH / f"{bench_family}"
        data["api"]["seed"] = SEED
        data["api"].pop("_target_")
        data["api"].pop("step_size")

        b = mfpbench.get(**data["api"])

        frac_stats = percentage_budget_maxima(b.table.copy(), b.value_metric)
        frac_stats.update(dict(
            is_mono=percentage_monotonic(
                b.table.copy(), b.value_metric, plot_mono=True, name=bench_name
            )
        ))
        frac_stats.update(dict(
            margin_share=margin_of_best_curve(
                b.table.copy(), b.value_metric, plot_margin=True, name=bench_name
            )
        ))
        # basic stats
        frac_stats.update(dict(
            num_configs=len(b.table.index.get_level_values(0).unique()),
            num_steps=len(b.table.index.get_level_values(1).unique()),
        ))
        print(path.name, frac_stats)

        all_data.update({bench_name: frac_stats})

        # plot all LCs
        plot_all_lcs(b.table.copy(), b.value_metric, bench_name)

    df = pd.DataFrame.from_dict(all_data, orient="index")
    df.to_parquet(BASE_PATH / "bench_summary.parquet")
