"""
This script assumes that pfns_hpo/utils/analyse_benchmarks.py has been run.
"""

import numpy as np
import pandas as pd
from pathlib import Path


SEED = 12345

PD1_issues = dict(
    low_steps=[  # df.num_steps.values < 10
        "pd1-tabular-cifar100_wideresnet_2048",
        "pd1-tabular-cifar10_wideresnet_2048",
        "pd1-tabular-fashion_simplecnn_2048",
        "pd1-tabular-mnist_simplecnn_2048",
    ],
    high_steps=[  # df.num_steps.values > 100
        "pd1-tabular-translate_xformertranslate_64-1",
        "pd1-tabular-translate_xformertranslate_64-2",
        "pd1-tabular-translate_xformertranslate_64-5",
        "pd1-tabular-translate_xformertranslate_64-10",
        "pd1-tabular-imagenet_resnet_512-1",
        "pd1-tabular-imagenet_resnet_512-2",
        "pd1-tabular-imagenet_resnet_256-1",
        "pd1-tabular-imagenet_resnet_256-2",
        "pd1-tabular-imagenet_resnet_1024-1",
    ],
    low_configs=[  # df.num_configs.values < 100
        "pd1-tabular-imagenet_resnet_1024-1",
        "pd1-tabular-imagenet_resnet_1024-2",
        "pd1-tabular-imagenet_resnet_1024-5",
        "pd1-tabular-imagenet_resnet_1024-10",
    ]
)


def split_df(df: pd.DataFrame) -> list[str]:
    split_idx = df.shape[0] // 2
    first_half = df.iloc[:split_idx].sort_values(by=["is_mono"], ascending=False)
    second_half = df.iloc[split_idx:].sort_values(by=["is_mono"], ascending=False)
    first_half = first_half.iloc[:first_half.shape[0] // 2]
    second_half = second_half.iloc[:second_half.shape[0] // 2]
    selection = np.concatenate((
        first_half.index.values, second_half.index.values
    )).tolist()
    return selection


if __name__ == "__main__":

    np.random.seed(SEED)

    VALIDATION_SET = dict(
        lcbench=[],
        lcbench_balanced=[],
        pd1=[],
    )

    BASE_PATH = Path(__file__).resolve().parent / ".." / "configs" / "benchmark"
    df = pd.read_parquet(BASE_PATH / "bench_summary.parquet")

    lcbench = df.loc[df.index.str.startswith("lcbench")]
    lcbench_balanced = lcbench.loc[lcbench.index.str.endswith("balanced")]
    lcbench = lcbench.loc[list(set(lcbench.index) - set(lcbench_balanced.index))]
    lcbench = lcbench.sort_values(by=["high", "mid", "low"], ascending=False)
    lcbench_balanced = lcbench_balanced.sort_values(
        by=["high", "mid", "low"], ascending=False
    )

    pd1 = df.loc[df.index.str.startswith("pd1")]
    pd1 = pd1.sort_values(by=["high", "mid", "low"], ascending=False)
    # TODO: do we drop these or not?
    # _idx = [_v for k, v in PD1_issues.items() for _v in v]
    _idx = PD1_issues["low_steps"]
    # _idx.extend(PD1_issues["low_configs"])
    pd1 = pd1.drop(index=_idx)

    taskset = df.loc[df.index.str.startswith("taskset")]
    taskset = taskset.sort_values(by=["high", "mid", "low"], ascending=False)

    # Splitting
    VALIDATION_SET["lcbench_balanced"] = np.random.choice(
        split_df(lcbench_balanced), size=5, replace=False
    )
    VALIDATION_SET["lcbench"] = [
        "-".join(_s.split("-")[:-1]) for _s in VALIDATION_SET["lcbench_balanced"]
    ]
    pd1_subset = df.loc[df.index.str.startswith("pd1")]
    VALIDATION_SET["pd1"] = pd1_subset.loc[
        ((pd1_subset.num_steps > 45) & (pd1_subset.num_steps < 55)).values
    ].index
    VALIDATION_SET["taskset"] = np.random.choice(
        split_df(taskset), size=5, replace=False
    )

    for k, v in VALIDATION_SET.items():
        for _v in v:
            print(f"* {_v}")
        print()
