from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Mapping, TypeVar
from typing_extensions import Self

import numpy as np

from mfpbench.benchmark import Config, Result
from mfpbench.metric import Metric
from mfpbench.yahpo.benchmark import YAHPOBenchmark

C = TypeVar("C", bound="IAMLConfig")
R = TypeVar("R", bound="IAMLResult")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class IAMLConfig(Config):
    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, Any],
        renames: Mapping[str, str] | None = None,
    ) -> Self:
        """Create from a dict or mapping object."""
        # We may have keys that are conditional and hence we need to flatten them
        config = {k.replace(".", "__"): v for k, v in d.items()}
        return super().from_dict(config, renames)

    def as_dict(self) -> dict[str, Any]:
        """Converts the config to a raw dictionary."""
        d = asdict(self)
        return {k.replace("__", "."): v for k, v in d.items() if v is not None}


@dataclass(frozen=True)  # type: ignore[misc]
class IAMLResult(Result[C, float]):
    default_value_metric: ClassVar[str] = "f1"
    default_value_metric_test: ClassVar[None] = None
    default_cost_metric: ClassVar[str] = "timetrain"
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "mmce": Metric(minimize=True, bounds=(0, np.inf)),
        "f1": Metric(minimize=False, bounds=(0, 1)),
        "auc": Metric(minimize=False, bounds=(0, 1)),
        "logloss": Metric(minimize=True, bounds=(0, np.inf)),
        "timetrain": Metric(minimize=True, bounds=(0, np.inf)),
        "timepredict": Metric(minimize=True, bounds=(0, np.inf)),
        "ramtrain": Metric(minimize=True, bounds=(0, np.inf)),
        "rammodel": Metric(minimize=True, bounds=(0, np.inf)),
        "rampredict": Metric(minimize=True, bounds=(0, np.inf)),
    }

    mmce: Metric.Value
    f1: Metric.Value
    auc: Metric.Value
    logloss: Metric.Value

    timetrain: Metric.Value
    timepredict: Metric.Value

    ramtrain: Metric.Value
    rammodel: Metric.Value
    rampredict: Metric.Value

    # Definitions taken from YAHPO-gym paper appendix
    # Whether to minimize is not really fully relevant
    # so these are not given a real Metric definition.
    mec: float  # main effect complexity of features
    ias: float  # Iteration stregth of features
    nf: float  # Number of features used


class IAMLBenchmark(YAHPOBenchmark[C, IAMLResult, float]):
    yahpo_result_type = IAMLResult
    # IAML class of benchmarks share train size as fidelity
    yahpo_fidelity_range = (0.03, 1.0, 0.05)
    yahpo_fidelity_name = "trainsize"
    yahpo_task_id_name = "task_id"
