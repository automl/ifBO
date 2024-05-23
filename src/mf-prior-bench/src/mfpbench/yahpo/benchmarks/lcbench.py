from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping

import numpy as np

from mfpbench.metric import Metric
from mfpbench.yahpo.benchmark import Config, Result, YAHPOBenchmark


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class LCBenchConfig(Config):
    """A LCBench Config.

    Note:
    ----
    For ``momentum``, the paper seems to suggest it's (0.1, 0.9) but the configspace
    says (0.1, 0.99), going with the code version.
    """

    batch_size: int  # [16, 512] int log
    learning_rate: float  # [1e-04, 0.1] float log
    momentum: float  # [0.1, 0.99] float, see note above
    weight_decay: float  # [1e-5, 0.1] float
    num_layers: int  # [1, 5] int
    max_units: int  # [64, 1024] int log
    max_dropout: float  # [0.0, 1.0] float


@dataclass(frozen=True)  # type: ignore[misc]
class LCBenchResult(Result[LCBenchConfig, int]):
    default_value_metric: ClassVar[str] = "val_balanced_accuracy"
    default_value_metric_test: ClassVar[str] = "test_balanced_accuracy"
    default_cost_metric: ClassVar[str] = "time"
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "val_accuracy": Metric(minimize=False, bounds=(0, 100)),
        "val_balanced_accuracy": Metric(minimize=False, bounds=(0, 1)),
        "val_cross_entropy": Metric(minimize=True, bounds=(0, np.inf)),
        "test_balanced_accuracy": Metric(minimize=False, bounds=(0, 1)),
        "test_cross_entropy": Metric(minimize=True, bounds=(0, np.inf)),
        "time": Metric(minimize=True, bounds=(0, np.inf)),
    }

    val_accuracy: Metric.Value
    val_cross_entropy: Metric.Value
    val_balanced_accuracy: Metric.Value

    test_cross_entropy: Metric.Value
    test_balanced_accuracy: Metric.Value

    time: Metric.Value  # unit?


class LCBenchBenchmark(YAHPOBenchmark):
    yahpo_fidelity_range = (1, 52, 1)
    yahpo_fidelity_name = "epoch"
    yahpo_config_type = LCBenchConfig
    yahpo_result_type = LCBenchResult
    yahpo_base_benchmark_name = "lcbench"
    yahpo_task_id_name = "OpenML_task_id"
    yahpo_has_conditionals = False
    yahpo_instances = (
        "3945",
        "7593",
        "34539",
        "126025",
        "126026",
        "126029",
        "146212",
        "167104",
        "167149",
        "167152",
        "167161",
        "167168",
        "167181",
        "167184",
        "167185",
        "167190",
        "167200",
        "167201",
        "168329",
        "168330",
        "168331",
        "168335",
        "168868",
        "168908",
        "168910",
        "189354",
        "189862",
        "189865",
        "189866",
        "189873",
        "189905",
        "189906",
        "189908",
        "189909",
    )
    """
    ```python exec="true" result="python"
    from mfpbench import LCBenchBenchmark
    print(LCBenchBenchmark.yahpo_instances)
    ```
    """
