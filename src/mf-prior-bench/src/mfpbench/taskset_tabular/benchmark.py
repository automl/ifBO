from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, TypeVar

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

from mfpbench.config import TabularConfig
from mfpbench.metric import Metric
from mfpbench.result import Result
from mfpbench.tabular import TabularBenchmark


def _get_raw_taskset_space(
    name: str,
    seed: int | None = None,
    *,
    optimizer: str,
) -> ConfigurationSpace:
    cs = ConfigurationSpace(name=name, seed=seed)
    cs.add_hyperparameters(
        [
            UniformFloatHyperparameter(
                "learning_rate",
                lower=1e-9,
                upper=10,
                log=True,
            ),
        ],
    )
    if optimizer.split("_")[0] in ["adam4p", "adam6p", "adam8p"]:
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "beta1",
                    lower=1e-4,
                    upper=1,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "beta2",
                    lower=1e-3,
                    upper=1,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "epsilon",
                    lower=1e-12,
                    upper=1000,
                    log=True,
                ),
            ],
        )
    if optimizer.split("_")[0] in ["adam6p", "adam8p"]:
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "l1",
                    lower=1e-9,
                    upper=10,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "l2",
                    lower=1e-9,
                    upper=10,
                    log=True,
                ),
            ],
        )
    if optimizer.split("_")[0] in ["adam8p"]:
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "linear_decay",
                    lower=1e-8,
                    upper=0.0001,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "exponential_decay",
                    lower=1e-6,
                    upper=1e-3,
                    log=True,
                ),
            ],
        )
    return cs


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig(TabularConfig):
    pass


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_1p(TaskSetTabularConfig):
    learning_rate: float


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_4p(TaskSetTabularConfig):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_6p(TaskSetTabularConfig):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    linear_decay: float
    exponential_decay: float


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_8p(TaskSetTabularConfig):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    linear_decay: float
    exponential_decay: float
    l1: float
    l2: float


C = TypeVar("C", bound=TaskSetTabularConfig)


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult(Result[C, int]):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(0, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }
    default_value_metric: ClassVar[str] = "valid1_loss"
    default_value_metric_test: ClassVar[str] = "test_loss"
    default_cost_metric: ClassVar[str] = "train_cost"

    train_loss: Metric.Value
    valid1_loss: Metric.Value
    valid2_loss: Metric.Value
    test_loss: Metric.Value
    train_cost: Metric.Value


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_adam1p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1737, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1702, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1707, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1707, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_adam4p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1802, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1767, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1770, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1770, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_adam6p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-3515, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-3480, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-3469, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-3469, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_adam8p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-3084, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-3057, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-3073, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-3073, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_nadamw_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1829, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1787, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1796, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1796, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_adam1p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-759, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-751, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-761, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-761, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_adam4p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-758, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-754, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-760, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-760, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_adam6p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-759, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-752, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-760, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-760, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_adam8p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-758, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-751, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-761, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-761, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_nadamw_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-759, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-752, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-761, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-761, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_adam1p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-985, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-951, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-959, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-959, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_adam4p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-999, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-942, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-961, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-961, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_adam6p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-994, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-939, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-953, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-953, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_adam8p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1008, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-959, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-966, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-966, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_nadamw_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1035, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-983, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-989, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-989, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_adam1p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1111, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1099, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1107, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1107, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_adam4p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1088, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1069, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1081, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1081, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_adam6p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1068, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1010, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1057, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1057, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_adam8p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1048, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1032, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1043, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1043, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_nadamw_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1140, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1122, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1133, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1133, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_adam1p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1339, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1256, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1326, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1326, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_adam4p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1306, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1080, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1292, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1292, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_adam6p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1304, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1053, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1289, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1289, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_adam8p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1247, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1148, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1231, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1231, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_nadamw_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1268, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1249, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1260, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1260, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_adam1p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1286, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1274, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1285, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1285, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_adam4p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1196, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1185, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1193, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1193, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_adam6p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1180, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1167, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1174, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1174, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_adam8p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1231, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1224, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1232, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1232, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_nadamw_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1293, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1287, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1293, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1293, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_adam1p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1515, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1506, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1516, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1516, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_adam4p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1529, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1516, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1525, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1525, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_adam6p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1335, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1323, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1330, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1330, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_adam8p_wide_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1454, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1446, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1454, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1454, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_nadamw_grid_1k(
    TaskSetTabularResult,
):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(-1383, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(-1374, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(-1380, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(-1380, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }


class TaskSetTabularBenchmark(
    TabularBenchmark[TaskSetTabularConfig, TaskSetTabularResult, int],
):
    """The taskset tabular benchmark.

    NOTE: The "epoch" that is used as the index is not actually the epoch
    but is used to keep it inline with other benchmarks. Please refer
    to the "step" column for the measurement of fidelity used. We can do
    so pretty safely for benchmarking as the steps are incremented in a uniform
    range.
    """

    task_ids: ClassVar[tuple[str, ...]] = (
        "Associative_GRU128_BS128_Pairs10_Tokens50",
        "Associative_GRU256_BS128_Pairs20_Tokens50",
        "Associative_LSTM128_BS128_Pairs10_Tokens50",
        "Associative_LSTM128_BS128_Pairs20_Tokens50",
        "Associative_LSTM128_BS128_Pairs5_Tokens20",
        "Associative_LSTM256_BS128_Pairs20_Tokens50",
        "Associative_LSTM256_BS128_Pairs40_Tokens100",
        "Associative_VRNN128_BS128_Pairs10_Tokens50",
        "Associative_VRNN256_BS128_Pairs20_Tokens50",
        "char_rnn_language_model_family",
        "conv_fc_family",
        "conv_pooling_family",
        "Copy_GRU128_BS128_Length20_Tokens10",
        "Copy_GRU256_BS128_Length40_Tokens50",
        "Copy_LSTM128_BS128_Length20_Tokens10",
        "Copy_LSTM128_BS128_Length20_Tokens20",
        "Copy_LSTM128_BS128_Length50_Tokens5",
        "Copy_LSTM128_BS128_Length5_Tokens10",
        "Copy_LSTM256_BS128_Length40_Tokens50",
        "Copy_VRNN128_BS128_Length20_Tokens10",
        "Copy_VRNN256_BS128_Length40_Tokens50",
        "FixedImageConvAE_cifar10_32x32x32x32x32_bs128",
        "FixedImageConvAE_cifar10_32x64x8x64x32_bs128",
        "FixedImageConvAE_mnist_32x32x32x32x32_bs128",
        "FixedImageConvAE_mnist_32x64x32x64x32_bs512",
        "FixedImageConvAE_mnist_32x64x8x64x32_bs128",
        "FixedImageConv_cifar100_32x64x128_FC64x32_tanh_variance_scaling_bs64",
        "FixedImageConv_cifar100_32x64x64_flatten_bs128",
        "FixedImageConv_cifar100_bn_32x64x128x128_bs128",
        "FixedImageConv_cifar10_32x64x128_flatten_FC64x32_tanh_he_bs8",
        "FixedImageConv_cifar10_32x64x128_flatten_FC64x32_tanh_variance_scaling_bs64",
        "FixedImageConv_cifar10_32x64x128_he_bs64",
        "FixedImageConv_cifar10_32x64x128_largenormal_bs64",
        "FixedImageConv_cifar10_32x64x128_normal_bs64",
        "FixedImageConv_cifar10_32x64x128_smallnormal_bs64",
        "FixedImageConv_cifar10_32x64x128x128x128_avg_he_bs64",
        "FixedImageConv_cifar10_32x64x64_bs128",
        "FixedImageConv_cifar10_32x64x64_fc_64_bs128",
        "FixedImageConv_cifar10_32x64x64_flatten_bs128",
        "FixedImageConv_cifar10_32x64x64_tanh_bs64",
        "FixedImageConv_cifar10_batchnorm_32x32x32x64x64_bs128",
        "FixedImageConv_cifar10_batchnorm_32x64x64_bs128",
        "FixedImageConv_coil10032x32_bn_32x64x128x128_bs128",
        "FixedImageConv_colorectalhistology32x32_32x64x64_flatten_bs128",
        "FixedImageConv_food10164x64_Conv_32x64x64_flatten_bs64",
        "FixedImageConv_food101_batchnorm_32x32x32x64x64_bs128",
        "FixedImageConv_mnist_32x64x64_fc_64_bs128",
        "FixedImageConv_sun39732x32_bn_32x64x128x128_bs128",
        "FixedImageConvVAE_cifar10_32x64x128x64x128x64x32_bs128",
        "FixedImageConvVAE_cifar10_32x64x128x64x128x64x32_bs512",
        "FixedImageConvVAE_cifar10_32x64x128x64x32_bs128",
        "FixedImageConvVAE_cifar10_64x128x256x128x256x128x64_bs128",
        "FixedImageConvVAE_mnist_32x32x32x32x32_bs128",
        "FixedImageConvVAE_mnist_32x64x32x64x32_bs128",
        "FixedImageConvVAE_mnist_64x128x128x128x64_bs128",
        "FixedLM_lm1b_patch128_GRU128_embed64_avg_bs128",
        "FixedLM_lm1b_patch128_GRU256_embed64_avg_bs128",
        "FixedLM_lm1b_patch128_GRU64_embed64_avg_bs128",
        "FixedLM_lm1b_patch128_LSTM128_embed64_avg_bs128",
        "FixedLM_lm1b_patch128_LSTM256_embed64_avg_bs128",
        "FixedMAF_cifar10_3layer_bs64",
        "FixedMAF_mnist_2layer_bs64",
        "FixedMAF_mnist_3layer_thin_bs64",
        "FixedMLPAE_cifar10_128x32x128_bs128",
        "FixedMLPAE_mnist_128x32x128_bs128",
        "FixedMLPAE_mnist_32x32x32_bs128",
        "FixedMLP_cifar10_BatchNorm_128x128x128_relu_bs128",
        "FixedMLP_cifar10_BatchNorm_64x64x64x64x64_relu_bs128",
        "FixedMLP_cifar10_ce_128x128x128_relu_bs128",
        "FixedMLP_cifar10_Dropout02_128x128_relu_bs128",
        "FixedMLP_cifar10_Dropout05_128x128_relu_bs128",
        "FixedMLP_cifar10_Dropout08_128x128_relu_bs128",
        "FixedMLP_cifar10_LayerNorm_128x128x128_relu_bs128",
        "FixedMLP_cifar10_LayerNorm_128x128x128_tanh_bs128",
        "FixedMLP_cifar10_mse_128x128x128_relu_bs128",
        "FixedMLP_food10132x32_ce_128x128x128_relu_bs128",
        "FixedMLP_food10132x32_mse_128x128x128_relu_bs128",
        "FixedMLP_mnist_ce_128x128x128_relu_bs128",
        "FixedMLP_mnist_mse_128x128x128_relu_bs128",
        "FixedMLPVAE_cifar101_128x128x32x128x128_bs128",
        "FixedMLPVAE_cifar101_128x32x128_bs128",
        "FixedMLPVAE_food10132x32_128x64x32x64x128_bs64",
        "FixedMLPVAE_mnist_128x128x8x128_bs128",
        "FixedMLPVAE_mnist_128x64x32x64x128_bs64",
        "FixedMLPVAE_mnist_128x8x128x128_bs128",
        "FixedNVP_mnist_2layer_bs64",
        "FixedNVP_mnist_3layer_thin_bs64",
        "FixedNVP_mnist_5layer_bs64",
        "FixedNVP_mnist_5layer_thin_bs64",
        "FixedNVP_mnist_9layer_thin_bs16",
        "FixedTextRNNClassification_imdb_patch128_LSTM128_avg_bs64",
        "FixedTextRNNClassification_imdb_patch128_LSTM128_bs64",
        "FixedTextRNNClassification_imdb_patch128_LSTM128_embed128_bs64",
        "FixedTextRNNClassification_imdb_patch32_GRU128_bs128",
        "FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128",
        "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_avg_bs128",
        "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_last_bs128",
        "FixedTextRNNClassification_imdb_patch32_LSTM128_bs128",
        "FixedTextRNNClassification_imdb_patch32_LSTM128_E128_bs128",
        "FixedTextRNNClassification_imdb_patch32_VRNN128_tanh_bs128",
        "FixedTextRNNClassification_imdb_patch32_VRNN64_relu_avg_bs128",
        "FixedTextRNNClassification_imdb_patch32_VRNN64_tanh_avg_bs128",
        "Imagenet32x30_FC_VAE_128x64x32x64x128_relu_bs256",
        "losg_tasks_family",
        "maf_family",
        "mlp_ae_family",
        "mlp_family",
        "mlp_vae_family",
        "Mnist_Conv_32x16x64_flatten_FC32_tanh_bs32",
        "nvp_family",
        "quadratic_family",
        "rnn_text_classification_family",
        "TwoD_Ackley",
        "TwoD_Beale",
        "TwoD_Bowl1",
        "TwoD_Bowl10",
        "TwoD_Bowl100",
        "TwoD_Bowl1000",
        "TwoD_Rosenbrock",
        "TwoD_StyblinskiTang",
        "word_rnn_language_model_family",
    )
    optimizers: ClassVar[tuple[str, ...]] = (
        "adam1p_wide_grid_1k",
        "adam4p_wide_grid_1k",
        "adam6p_wide_grid_1k",
        "adam8p_wide_grid_1k",
        "nadamw_grid_1k",
    )
    _optimizer_map: ClassVar[Mapping[str, str]] = {
        "adam1p": "adam1p_wide_grid_1k",
        "adam4p": "adam4p_wide_grid_1k",
        "adam6p": "adam6p_wide_grid_1k",
        "adam8p": "adam8p_wide_grid_1k",
        "nadamw": "nadamw_grid_1k",
    }
    illegal_combinations: ClassVar[set[tuple[str, str]]] = {
        ("char_rnn_language_model_family", "adam1p_wide_grid_1k"),
        ("char_rnn_language_model_family", "adam4p_wide_grid_1k"),
        ("char_rnn_language_model_family", "adam6p_wide_grid_1k"),
        ("char_rnn_language_model_family", "adam8p_wide_grid_1k"),
        ("char_rnn_language_model_family", "nadamw_grid_1k"),
        ("conv_fc_family", "adam1p_wide_grid_1k"),
        ("conv_fc_family", "adam4p_wide_grid_1k"),
        ("conv_fc_family", "adam6p_wide_grid_1k"),
        ("conv_fc_family", "adam8p_wide_grid_1k"),
        ("conv_fc_family", "nadamw_grid_1k"),
        ("conv_pooling_family", "adam1p_wide_grid_1k"),
        ("conv_pooling_family", "adam4p_wide_grid_1k"),
        ("conv_pooling_family", "adam6p_wide_grid_1k"),
        ("conv_pooling_family", "adam8p_wide_grid_1k"),
        ("conv_pooling_family", "nadamw_grid_1k"),
        ("losg_tasks_family", "adam1p_wide_grid_1k"),
        ("losg_tasks_family", "adam4p_wide_grid_1k"),
        ("losg_tasks_family", "adam6p_wide_grid_1k"),
        ("losg_tasks_family", "adam8p_wide_grid_1k"),
        ("losg_tasks_family", "nadamw_grid_1k"),
        ("maf_family", "adam1p_wide_grid_1k"),
        ("maf_family", "adam4p_wide_grid_1k"),
        ("maf_family", "adam6p_wide_grid_1k"),
        ("maf_family", "adam8p_wide_grid_1k"),
        ("maf_family", "nadamw_grid_1k"),
        ("mlp_ae_family", "adam1p_wide_grid_1k"),
        ("mlp_ae_family", "adam4p_wide_grid_1k"),
        ("mlp_ae_family", "adam6p_wide_grid_1k"),
        ("mlp_ae_family", "adam8p_wide_grid_1k"),
        ("mlp_ae_family", "nadamw_grid_1k"),
        ("mlp_family", "adam1p_wide_grid_1k"),
        ("mlp_family", "adam4p_wide_grid_1k"),
        ("mlp_family", "adam6p_wide_grid_1k"),
        ("mlp_family", "adam8p_wide_grid_1k"),
        ("mlp_family", "nadamw_grid_1k"),
        ("mlp_vae_family", "adam1p_wide_grid_1k"),
        ("mlp_vae_family", "adam4p_wide_grid_1k"),
        ("mlp_vae_family", "adam6p_wide_grid_1k"),
        ("mlp_vae_family", "adam8p_wide_grid_1k"),
        ("mlp_vae_family", "nadamw_grid_1k"),
        ("nvp_family", "adam1p_wide_grid_1k"),
        ("nvp_family", "adam4p_wide_grid_1k"),
        ("nvp_family", "adam6p_wide_grid_1k"),
        ("nvp_family", "adam8p_wide_grid_1k"),
        ("nvp_family", "nadamw_grid_1k"),
        ("quadratic_family", "adam1p_wide_grid_1k"),
        ("quadratic_family", "adam4p_wide_grid_1k"),
        ("quadratic_family", "adam6p_wide_grid_1k"),
        ("quadratic_family", "adam8p_wide_grid_1k"),
        ("quadratic_family", "nadamw_grid_1k"),
        ("rnn_text_classification_family", "adam1p_wide_grid_1k"),
        ("rnn_text_classification_family", "adam4p_wide_grid_1k"),
        ("rnn_text_classification_family", "adam6p_wide_grid_1k"),
        ("rnn_text_classification_family", "adam8p_wide_grid_1k"),
        ("rnn_text_classification_family", "nadamw_grid_1k"),
        ("word_rnn_language_model_family", "adam1p_wide_grid_1k"),
        ("word_rnn_language_model_family", "adam4p_wide_grid_1k"),
        ("word_rnn_language_model_family", "adam6p_wide_grid_1k"),
        ("word_rnn_language_model_family", "adam8p_wide_grid_1k"),
        ("word_rnn_language_model_family", "nadamw_grid_1k"),
    }
    _optimizer_config_map: ClassVar[Mapping[str, type[TaskSetTabularConfig]]] = {
        "adam1p_wide_grid_1k": TaskSetTabularConfig_1p,
        "adam4p_wide_grid_1k": TaskSetTabularConfig_4p,
        "adam6p_wide_grid_1k": TaskSetTabularConfig_6p,
        "adam8p_wide_grid_1k": TaskSetTabularConfig_8p,
        "nadamw_grid_1k": TaskSetTabularConfig_1p,
    }
    _result_map: ClassVar[Mapping[tuple[str, str], type[TaskSetTabularResult]]] = {
        (
            "FixedMAF_mnist_2layer_bs64",
            "adam1p_wide_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_adam1p_wide_grid_1k,
        (
            "FixedMAF_mnist_2layer_bs64",
            "adam4p_wide_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_adam4p_wide_grid_1k,
        (
            "FixedMAF_mnist_2layer_bs64",
            "adam6p_wide_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_adam6p_wide_grid_1k,
        (
            "FixedMAF_mnist_2layer_bs64",
            "adam8p_wide_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_adam8p_wide_grid_1k,
        (
            "FixedMAF_mnist_2layer_bs64",
            "nadamw_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_2layer_bs64_nadamw_grid_1k,
        (
            "FixedMAF_mnist_3layer_thin_bs64",
            "adam1p_wide_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_adam1p_wide_grid_1k,
        (
            "FixedMAF_mnist_3layer_thin_bs64",
            "adam4p_wide_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_adam4p_wide_grid_1k,
        (
            "FixedMAF_mnist_3layer_thin_bs64",
            "adam6p_wide_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_adam6p_wide_grid_1k,
        (
            "FixedMAF_mnist_3layer_thin_bs64",
            "adam8p_wide_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_adam8p_wide_grid_1k,
        (
            "FixedMAF_mnist_3layer_thin_bs64",
            "nadamw_grid_1k",
        ): TaskSetTabularResult_FixedMAF_mnist_3layer_thin_bs64_nadamw_grid_1k,
        (
            "FixedNVP_mnist_2layer_bs64",
            "adam1p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_adam1p_wide_grid_1k,
        (
            "FixedNVP_mnist_2layer_bs64",
            "adam4p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_adam4p_wide_grid_1k,
        (
            "FixedNVP_mnist_2layer_bs64",
            "adam6p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_adam6p_wide_grid_1k,
        (
            "FixedNVP_mnist_2layer_bs64",
            "adam8p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_adam8p_wide_grid_1k,
        (
            "FixedNVP_mnist_2layer_bs64",
            "nadamw_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_2layer_bs64_nadamw_grid_1k,
        (
            "FixedNVP_mnist_3layer_thin_bs64",
            "adam1p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_adam1p_wide_grid_1k,
        (
            "FixedNVP_mnist_3layer_thin_bs64",
            "adam4p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_adam4p_wide_grid_1k,
        (
            "FixedNVP_mnist_3layer_thin_bs64",
            "adam6p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_adam6p_wide_grid_1k,
        (
            "FixedNVP_mnist_3layer_thin_bs64",
            "adam8p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_adam8p_wide_grid_1k,
        (
            "FixedNVP_mnist_3layer_thin_bs64",
            "nadamw_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_3layer_thin_bs64_nadamw_grid_1k,
        (
            "FixedNVP_mnist_5layer_bs64",
            "adam1p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_adam1p_wide_grid_1k,
        (
            "FixedNVP_mnist_5layer_bs64",
            "adam4p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_adam4p_wide_grid_1k,
        (
            "FixedNVP_mnist_5layer_bs64",
            "adam6p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_adam6p_wide_grid_1k,
        (
            "FixedNVP_mnist_5layer_bs64",
            "adam8p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_adam8p_wide_grid_1k,
        (
            "FixedNVP_mnist_5layer_bs64",
            "nadamw_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_bs64_nadamw_grid_1k,
        (
            "FixedNVP_mnist_5layer_thin_bs64",
            "adam1p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_adam1p_wide_grid_1k,
        (
            "FixedNVP_mnist_5layer_thin_bs64",
            "adam4p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_adam4p_wide_grid_1k,
        (
            "FixedNVP_mnist_5layer_thin_bs64",
            "adam6p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_adam6p_wide_grid_1k,
        (
            "FixedNVP_mnist_5layer_thin_bs64",
            "adam8p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_adam8p_wide_grid_1k,
        (
            "FixedNVP_mnist_5layer_thin_bs64",
            "nadamw_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_5layer_thin_bs64_nadamw_grid_1k,
        (
            "FixedNVP_mnist_9layer_thin_bs16",
            "adam1p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_adam1p_wide_grid_1k,
        (
            "FixedNVP_mnist_9layer_thin_bs16",
            "adam4p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_adam4p_wide_grid_1k,
        (
            "FixedNVP_mnist_9layer_thin_bs16",
            "adam6p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_adam6p_wide_grid_1k,
        (
            "FixedNVP_mnist_9layer_thin_bs16",
            "adam8p_wide_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_adam8p_wide_grid_1k,
        (
            "FixedNVP_mnist_9layer_thin_bs16",
            "nadamw_grid_1k",
        ): TaskSetTabularResult_FixedNVP_mnist_9layer_thin_bs16_nadamw_grid_1k,
    }

    def __init__(
        self,
        task_id: str,
        optimizer: str,
        datadir: str | Path | None = None,
        *,
        seed: int | None = None,
        prior: str | Path | TaskSetTabularConfig | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
        na_with_inf: bool = True,
    ) -> None:
        """Initialize a taskset tabular benchmark.

        Args:
            task_id: The task id to use.
            optimizer: The optimizer to use.
            datadir: The directory to use for the data.
            remove_constants: Whether or not to remove constants from the
                config space.
            seed: The seed to use for the benchmark.
            prior: The prior to use for the benchmark.
            perturb_prior: The perturbation to use for the prior.
            value_metric: The value metric to use for the benchmark.
            value_metric_test: The test value metric to use for the benchmark.
            na_with_inf: Whether or not to replace NaNs with inf.
            cost_metric: The cost metric to use for the benchmark.
        """
        cls = self.__class__
        if task_id not in cls.task_ids:
            raise ValueError(
                f"Unknown task {task_id}, must be one of {cls.task_ids}",
            )
        if optimizer not in cls.optimizers and optimizer not in cls._optimizer_map:
            raise ValueError(
                f"Unknown task {optimizer}, must be one of {cls.optimizers}",
                f"Or {optimizer}, must be one of {list(cls._optimizer_map.keys())}",
            )
        if optimizer in cls._optimizer_map:
            optimizer = cls._optimizer_map[optimizer]

        if (task_id, optimizer) in cls.illegal_combinations:
            raise ValueError(
                f"These are the illegal combinations: {cls.illegal_combinations}.",
                f"\nCannot use task {task_id} with optimizer {optimizer}.",
            )

        if datadir is None:
            from mfpbench.setup_benchmark import TaskSetabularSource

            datadir = TaskSetabularSource.default_location()

        name = f"{task_id}-{optimizer}"
        filename = f"{name}_10000_replica5.parquet"
        table_path = Path(datadir) / filename
        if not table_path.exists():
            raise FileNotFoundError(
                f"Could not find table {table_path}."
                f"`python -m mfpbench download --status --data-dir {datadir}",
            )
        table = pd.read_parquet(table_path)
        space = _get_raw_taskset_space(
            name=name,
            seed=seed,
            optimizer=optimizer,
        )
        config_type = cls._optimizer_config_map[optimizer]
        result_type = cls._result_map.get((task_id, optimizer), TaskSetTabularResult)

        # renaming config_id for consistency across tabular benchmarks
        table.index = table.index.set_names("id", level=0)

        if na_with_inf:
            table = table.fillna(np.inf)

        super().__init__(
            table=table,  # type: ignore
            name=name,
            id_key="id",
            fidelity_key="epoch",
            result_type=result_type,
            config_type=config_type,  # type: ignore
            info_keys=["step"],
            value_metric=value_metric,
            value_metric_test=value_metric_test,
            cost_metric=cost_metric,
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
        )

    def _normalize_each_curve(
        self,
        df: pd.DataFrame,
        metrics: list[str],
    ) -> pd.DataFrame:
        """Normalizing each curve to [0, 1] and handling NaNs.

        Section 3.3 from https://arxiv.org/abs/2002.11887
        """
        global_min_losses = df[metrics].min()

        def _normalize_metrics(config_frame_column: pd.DataFrame) -> pd.DataFrame:
            # Get's one column at a time, we only normalize if it's a metric
            if config_frame_column.name not in metrics:
                return config_frame_column

            # normalize as (x - min) / (max - min)
            _min = global_min_losses[config_frame_column.name]
            _max = config_frame_column.max()
            normalized = config_frame_column.subtract(_min).divide(_max - _min)
            return normalized.fillna(np.inf).clip(0, 1)

        return df.groupby("id").transform(_normalize_metrics)

    def _remove_zero_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops the loss curve at step 0, that is, at initialization"""
        unique_ids = df.index.get_level_values(0).unique()
        # check if step=0 exists for all unique IDs
        step_zero_exists_for_all = sum(df["step"] == 0) == len(unique_ids)
        if step_zero_exists_for_all is False:
            return df

        # dropping all rows with step as 0
        df = df.reset_index()
        df = df.drop(index=df.loc[df["step"] == 0].index)
        # reindexing to enumerate fidelity steps
        df["epoch"] = df["epoch"] - 1
        return df.set_index(["id", "epoch"])
