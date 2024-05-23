from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Mapping, TypeVar
from typing_extensions import Self

import numpy as np

from mfpbench.benchmark import Config, Result
from mfpbench.metric import Metric
from mfpbench.yahpo.benchmark import YAHPOBenchmark

C = TypeVar("C", bound="RBV2Config")
R = TypeVar("R", bound="RBV2Result")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class RBV2Config(Config):
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
class RBV2Result(Result[C, float]):
    default_value_metric: ClassVar[str] = "bac"
    default_cost_metric: ClassVar[str] = "timetrain"
    default_cost_metric_test: ClassVar[None] = None
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "acc": Metric(minimize=False, bounds=(0, 1)),
        "bac": Metric(minimize=False, bounds=(0, 1)),
        "auc": Metric(minimize=False, bounds=(0, 1)),
        "brier": Metric(minimize=True, bounds=(0, 1)),
        "f1": Metric(minimize=False, bounds=(0, 1)),
        "logloss": Metric(minimize=True, bounds=(0, np.inf)),
        "timetrain": Metric(minimize=True, bounds=(0, np.inf)),
        "timepredict": Metric(minimize=True, bounds=(0, np.inf)),
        "memory": Metric(minimize=True, bounds=(0, np.inf)),
    }

    acc: Metric.Value
    bac: Metric.Value
    auc: Metric.Value
    brier: Metric.Value
    f1: Metric.Value
    logloss: Metric.Value

    timetrain: Metric.Value
    timepredict: Metric.Value

    memory: Metric.Value


class RBV2Benchmark(YAHPOBenchmark[C, RBV2Result, float]):
    # RVB2 class of benchmarks share train size as fidelity
    yahpo_config_type: type[C]
    yahpo_result_type = RBV2Result
    yahpo_fidelity_range = (0.03, 1.0, 0.05)
    yahpo_fidelity_name = "trainsize"
    yahpo_task_id_name = "task_id"

    # We have to specify a repl number, not sure what it is but YAHPO gym fix it to 10
    yahpo_forced_remove_hps: ClassVar[Mapping[str, int]] = {"repl": 10}
