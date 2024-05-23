"""Notes.

* Config 9075 for ImageNet w/ 200 epochs is missing train and test loss from epoch 68
    onwards. It was the only config of all of nb201 with missing data so we drop it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping

import numpy as np
import pandas as pd
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace

from mfpbench.metric import Metric
from mfpbench.result import Result
from mfpbench.setup_benchmark import NB201TabularSource
from mfpbench.tabular import TabularBenchmark, TabularConfig


def _raw_space(name: str, *, seed: int | None = None) -> ConfigurationSpace:
    choices = [
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    cs = ConfigurationSpace(name=name, seed=seed)
    cs.add_hyperparameters(
        [
            CategoricalHyperparameter("edge_0_1", choices=choices.copy()),
            CategoricalHyperparameter("edge_0_2", choices=choices.copy()),
            CategoricalHyperparameter("edge_0_3", choices=choices.copy()),
            CategoricalHyperparameter("edge_1_2", choices=choices.copy()),
            CategoricalHyperparameter("edge_1_3", choices=choices.copy()),
            CategoricalHyperparameter("edge_2_3", choices=choices.copy()),
        ],
    )
    return cs


@dataclass(frozen=True)  # type: ignore[misc]
class NB201Result(Result):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "train_accuracy": Metric(minimize=False, bounds=(0, 100)),
        "train_per_time": Metric(minimize=True, bounds=(0, np.inf)),
        "train_all_time": Metric(minimize=True, bounds=(0, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "test_accuracy": Metric(minimize=False, bounds=(0, 100)),
        "test_per_time": Metric(minimize=True, bounds=(0, np.inf)),
        "test_all_time": Metric(minimize=True, bounds=(0, np.inf)),
    }
    default_value_metric: ClassVar[str] = "train_accuracy"
    default_value_metric_test: ClassVar[str] = "test_accuracy"
    default_cost_metric: ClassVar[str] = "train_all_time"

    train_loss: Metric.Value
    train_accuracy: Metric.Value
    train_per_time: Metric.Value
    train_all_time: Metric.Value
    test_loss: Metric.Value
    test_accuracy: Metric.Value
    test_per_time: Metric.Value
    test_all_time: Metric.Value


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class NB201Config(TabularConfig):
    edge_0_1: Literal[
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    edge_0_2: Literal[
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    edge_0_3: Literal[
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    edge_1_2: Literal[
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    edge_1_3: Literal[
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    edge_2_3: Literal[
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]


class NB201TabularBenchmark(TabularBenchmark[NB201Config, NB201Result, int]):
    task_ids: ClassVar[tuple[str, ...]] = (
        "ImageNet16-120",
        "cifar10",
        "cifar10-valid",
        "cifar100",
    )
    max_epochs: ClassVar[tuple[int, int]] = (12, 200)

    def __init__(
        self,
        task_id: str,
        max_epoch: int,
        datadir: str | Path | None = None,
        *,
        seed: int | None = None,
        prior: str | Path | NB201Config | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
    ) -> None:
        """Initialize the benchmark.

        Args:
            task_id: The task to benchmark on.
            max_epoch: Use the `12` or `200` epoch version of the dataset.
            datadir: The directory to look for the data in. If `None`, uses the default
                download directory.
            seed: The seed to use.
            prior: The prior to use for the benchmark. If None, no prior is used.
                If a str, will check the local location first for a prior
                specific for this benchmark, otherwise assumes it to be a Path.
                If a Path, will load the prior from the path.
                If a Mapping, will be used directly.
            perturb_prior: If not None, will perturb the prior by this amount.
                For numericals, this is interpreted as the standard deviation of a
                normal distribution while for categoricals, this is interpreted
                as the probability of swapping the value for a random one.
            value_metric: The metric to use for this benchmark. Uses
                the default metric from the Result if None.
            value_metric_test: The test metric to use for this benchmark. Uses
                the default metric from the Result if None.
            cost_metric: The cost to use for this benchmark. Uses
                the default cost from the Result if None.
        """
        cls = self.__class__

        if task_id not in cls.task_ids:
            raise ValueError(f"Unknown task {task_id}, must be one of {cls.task_ids}")

        if datadir is None:
            datadir = NB201TabularSource.default_location()

        name = f"nb201_{task_id}_{max_epoch}"
        table_path = Path(datadir) / f"{name}.parquet"
        if not table_path.exists():
            raise FileNotFoundError(
                f"Could not find table {table_path}."
                f"`python -m mfpbench download --status --data-dir {datadir}",
            )

        self.task_id = task_id
        self.datadir = Path(datadir) if isinstance(datadir, str) else datadir
        space = _raw_space(name=name, seed=seed)
        table = pd.read_parquet(table_path)
        super().__init__(
            table=table,  # type: ignore
            name=name,
            id_key="config_id",
            fidelity_key="epoch",
            result_type=NB201Result,
            config_type=NB201Config,
            value_metric=value_metric,
            value_metric_test=value_metric_test,
            cost_metric=cost_metric,
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
        )
