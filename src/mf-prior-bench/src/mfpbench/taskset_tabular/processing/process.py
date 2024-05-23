"""Download and basic processing code source.

https://github.com/releaunifreiburg/DPL/blob/main/python_scripts/download_task_set_data.py
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from itertools import chain, product
from pathlib import Path
from typing import TypeVar
from typing_extensions import Self

import numpy as np
import pandas as pd

from mfpbench.taskset_tabular.benchmark import TaskSetTabularBenchmark

D = TypeVar("D")
R = TypeVar("R")


@dataclass
class Curves:
    """Curves for a given task."""

    train: list[np.ndarray]
    valid1: list[np.ndarray]
    valid2: list[np.ndarray]
    test: list[np.ndarray]

    def __post_init__(self) -> None:
        all_curves = chain(self.train, self.valid1, self.valid2, self.test)
        first_curve = next(all_curves)
        length = len(first_curve)
        if not all(len(x) == length for x in all_curves):
            raise ValueError("All curves must be same length")

    def df(self) -> pd.DataFrame:
        """Returns a dataframe with the curves."""
        return (
            pd.DataFrame(
                {
                    "train_loss": self.train,
                    "valid1_loss": self.valid1,
                    "valid2_loss": self.valid2,
                    "test_loss": self.test,
                    "config_id": np.arange(len(self.train)),
                },
            )
            .astype({"config_id": "Int32"})
            .set_index("config_id")
        )


HP_BASE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/task_set/optimizers/configs"


def hyperparameter_from_optimizer(
    optimizer: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Returns the hyperparameter name from the optimizer name.


    Args:
        optimizer: The optimizer name
        cache_dir: If not None, will use this directory as a cache to save and load

    Returns:
        A dataframe with optimizer and seed as the index, and hyperparameters as
        columns.

                          | HP1 | HP2 | HP3 | ... |
        optimizer  | seed | ----|-----|-----| --- |
                   |      |     |     |     |     |
                   |      |     |     |     |     |
                   |      |     |     |     |     |
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{optimizer.strip('_1k')}-hps.parquet"

    if cache_path is not None and cache_path.exists():
        print("Loading cache from", cache_path)  # noqa: T201
        return pd.read_parquet(cache_path)

    # Get the hyperparameters for the optimizer
    path = f"{HP_BASE_URL}/{optimizer.strip('_1k')}.json"
    try:
        print("Downloading from", path)  # noqa: T201
        with urllib.request.urlopen(path) as content:  # noqa: S310
            configs = json.loads(content.read())
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Failed for {optimizer} hyperparameters") from e

    # v[1] just contains the optimizer name again
    configs = {k: v[0] for k, v in configs.items()}
    df = pd.DataFrame.from_dict(configs, orient="index")

    # The index names are of the form {optimizer}_seed{seed}
    # We assign these to new columns
    # As there is a one to one mapping of seed to config, I assume
    # we can use it as a config_id
    df[["optimizer", "config_id"]] = df.apply(
        lambda row: row.name.split("_seed"),
        axis=1,
        result_type="expand",
    ).astype({0: "category", 1: "Int32"})

    df = df.convert_dtypes().set_index(["optimizer", "config_id"]).sort_index()
    if cache_path is not None:
        print("Saving to", cache_path)  # noqa: T201
        df.to_parquet(cache_path)

    return df


@dataclass
class DataPack:
    """Data pack for a task."""

    task: str
    optimizer: str
    df: pd.DataFrame | None = None

    @classmethod
    def list_tasks(cls, cache_dir: Path | None = None) -> list[str]:
        """Returns a list of task names.

        Args:
            cache_dir: If not None, will use this directory as a cache to save and load
                the task names to this directory.

        Returns:
            A dataframe, indexed by the full task name, with columns for the task name
            and seed. If the dataset doesn't have a seed, it will be set to pd.NA.
        """

        def _download() -> list[str]:
            from tensorflow.compat.v2.io.gfile import GFile  # type: ignore

            with GFile("gs://gresearch/task_set_data/task_names.txt") as f:
                return sorted(f.read().strip().split("\n"))

        if cache_dir is None:
            return _download()

        cache_dir.mkdir(parents=True, exist_ok=True)
        CACHE_FILE = cache_dir / "taskset_task_names.txt"
        if CACHE_FILE.exists():
            return sorted(CACHE_FILE.read_text().strip().split("\n"))

        full_names = _download()
        CACHE_FILE.write_text("\n".join(full_names))
        return full_names

    @classmethod
    def load(cls, task: str, optimizer: str, cache_dir: Path | None = None) -> Self:
        """Loads the learning curves for the given task and opt_set_name.

        Args:
            task: Name of the task
            optimizer: Name of the optimizer set
            cache_dir: If not None, will use this directory as a cache to save and load

        Returns:
            A tuple of (optimizer names X seed, x data points, and y data points)
        """
        from tensorflow.compat.v2.io.gfile import GFile  # type: ignore

        name = f"{optimizer}_10000_replica5"

        cache_path = (
            cache_dir / f"{task}-{name}.parquet" if cache_dir is not None else None
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                print("Loading cache from", cache_path)  # noqa: T201
                df = pd.read_parquet(cache_path)
                return cls(task=task, optimizer=optimizer, df=df)

        # Get the curves
        path = "/".join(["gs://gresearch/task_set_data", task, f"{name}.npz"])
        print(f"Downloading curves for {task} {optimizer} from {path}")  # noqa: T201
        try:
            with GFile(path, "rb") as file:
                cc = np.load(file)
                steps = cc["xs"]
                ys = cc["ys"]
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed for {task}-{optimizer} curves") from e

        hps = hyperparameter_from_optimizer(optimizer, cache_dir=cache_dir)
        hps = hps.droplevel("optimizer")  # Don't need the optimizer name in the index

        curves = cls._extract_curves(ys)
        curve_df = curves.df()

        # The both join on the `"config_id"` index
        joint_table = hps.join(curve_df)

        # Right now, the "train", "valid1", "valid2", "test" columns are lists of size
        # 51, which indicates it's loss as steps.
        # We assign the "step" column to match this before we explode the lists, one
        # full `"step"` list for each row.
        joint_table["step"] = [steps] * len(joint_table)

        # We also assign an "epoch" to be able to be inline with other benchmarks
        # NOTE: This is not really an epoch and is just a proxy
        joint_table["epoch"] = [np.arange(1, len(steps) + 1)] * len(joint_table)


        # There's also no cost information so we just use the step as the cost
        joint_table["train_cost"] = joint_table["step"].copy()
        # NOTE: Use train_cost as `cost` will conflict with the Result class

        # Now we explode everything out to get the following table. The HP columns
        # will repeat mostly but that's how we need in later in the benchmark.
        #
        #                   | HP1 | HP2 | HP3 | ... | train | valid1 | valid2  | test | cost |  # noqa: E501
        # config_id | epoch |
        #     0     | 1     |
        #           | 2     |
        #           | ...   |
        #           | 51    |
        #     1     | 1     |
        #           | 2     |
        #           | ...   |
        #           | 51    |
        df = (
            joint_table.explode(
                [
                    "train_loss",
                    "valid1_loss",
                    "valid2_loss",
                    "test_loss",
                    "train_cost",
                    "step",
                    "epoch",
                ],
            )
            .reset_index()
            .convert_dtypes()
            .set_index(["config_id", "epoch"])
            .sort_index()
        )
        if cache_path is not None:
            print("Saving df to", cache_path)  # noqa: T201
            df.to_parquet(cache_path)

        return cls(task=task, optimizer=optimizer, df=df)

    @staticmethod
    def _extract_curves(y: np.ndarray) -> Curves:
        """Returns the curve, matching the order of hyperparameters as
        gotten from hyperparamter dataframe.
        """
        n_seeds = y.shape[1]

        def _mean_curve(c_i: int) -> list[np.ndarray]:
            return [
                np.mean([y[hp_i, seed_i, :, c_i] for seed_i in range(n_seeds)], axis=0)
                for hp_i in range(y.shape[0])
            ]

        return Curves(
            train=_mean_curve(0),
            valid1=_mean_curve(1),
            valid2=_mean_curve(2),
            test=_mean_curve(2),
        )


def process_taskset(output_dir: Path, workers: int = 1) -> None:
    """Process the taskset data."""
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.Parallel(n_jobs=workers)(
        joblib.delayed(hyperparameter_from_optimizer)(optimizer, cache_dir=output_dir)
        for optimizer in TaskSetTabularBenchmark.optimizers
    )
    # Make sure all the configs are downloaded onto disk
    combinations = [
        (task, optimizer)
        for task, optimizer in product(
            TaskSetTabularBenchmark.task_ids,
            TaskSetTabularBenchmark.optimizers,
        )
        if (task, optimizer) not in TaskSetTabularBenchmark.illegal_combinations
    ]
    joblib.Parallel(n_jobs=workers)(
        joblib.delayed(DataPack.load)(task, optimizer, cache_dir=output_dir)
        for task, optimizer in combinations
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare hyperparameter candidates from the taskset task",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "The output directory where the validation curves"
            " and hyperparameter configurations will be saved"
        ),
        type=Path,
        default=Path("./taskset"),
    )
    parser.add_argument(
        "--workers",
        help="Number of workers to use for parallel processing",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    process_taskset(output_dir=args.output_dir, workers=args.workers)
# end of file
