from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Mapping
from typing_extensions import Literal, Self

import numpy as np

from mfpbench.benchmark import Config, Result
from mfpbench.metric import Metric
from mfpbench.yahpo.benchmark import YAHPOBenchmark

ChoicesT = Literal[
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

Choices = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

_hp_name_extension = "NetworkSelectorDatasetInfo_COLON_darts_COLON_"


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class NB301Config(Config):
    edge_normal_0: ChoicesT
    edge_normal_1: ChoicesT

    edge_reduce_0: ChoicesT
    edge_reduce_1: ChoicesT

    inputs_node_reduce_3: Literal["0_1", "0_2", "1_2"]
    inputs_node_reduce_4: Literal["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]
    inputs_node_reduce_5: Literal[
        "0_1",
        "0_2",
        "0_3",
        "0_4",
        "1_2",
        "1_3",
        "1_4",
        "2_3",
        "2_4",
        "3_4",
    ]

    inputs_node_normal_3: Literal["0_1", "0_2", "1_2"]
    inputs_node_normal_4: Literal["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]
    inputs_node_normal_5: Literal[
        "0_1",
        "0_2",
        "0_3",
        "0_4",
        "1_2",
        "1_3",
        "1_4",
        "2_3",
        "2_4",
        "3_4",
    ]

    edge_normal_2: ChoicesT | None = None
    edge_normal_3: ChoicesT | None = None
    edge_normal_4: ChoicesT | None = None
    edge_normal_5: ChoicesT | None = None
    edge_normal_6: ChoicesT | None = None
    edge_normal_7: ChoicesT | None = None
    edge_normal_8: ChoicesT | None = None
    edge_normal_9: ChoicesT | None = None
    edge_normal_10: ChoicesT | None = None
    edge_normal_11: ChoicesT | None = None
    edge_normal_12: ChoicesT | None = None
    edge_normal_13: ChoicesT | None = None

    edge_reduce_2: ChoicesT | None = None
    edge_reduce_3: ChoicesT | None = None
    edge_reduce_4: ChoicesT | None = None
    edge_reduce_5: ChoicesT | None = None
    edge_reduce_6: ChoicesT | None = None
    edge_reduce_7: ChoicesT | None = None
    edge_reduce_8: ChoicesT | None = None
    edge_reduce_9: ChoicesT | None = None
    edge_reduce_10: ChoicesT | None = None
    edge_reduce_11: ChoicesT | None = None
    edge_reduce_12: ChoicesT | None = None
    edge_reduce_13: ChoicesT | None = None

    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, Any],
        renames: Mapping[str, str] | None = None,
    ) -> Self:
        """Create from a dict or mapping object."""
        # We may have keys that are conditional and hence we need to flatten them
        config = {k.replace(_hp_name_extension, ""): v for k, v in d.items()}
        return super().from_dict(config, renames)

    def as_dict(self) -> dict[str, Any]:
        """Converts the config to a raw dictionary."""
        return {
            _hp_name_extension + k: v for k, v in asdict(self).items() if v is not None
        }


@dataclass(frozen=True)  # type: ignore[misc]
class NB301Result(Result[NB301Config, int]):
    default_value_metric: ClassVar[str] = "val_accuracy"
    default_value_metric_test: ClassVar[None] = None
    default_cost_metric: ClassVar[str] = "runtime"
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "runtime": Metric(minimize=True, bounds=(0, np.inf)),
        "val_accuracy": Metric(minimize=False, bounds=(0, 1)),
    }

    runtime: Metric.Value  # unit?
    val_accuracy: Metric.Value


class NB301Benchmark(YAHPOBenchmark):
    yahpo_fidelity_name = "epoch"
    yahpo_fidelity_range = (1, 98, 1)
    yahpo_config_type = NB301Config
    yahpo_result_type = NB301Result
    yahpo_has_conditionals = True
    yahpo_base_benchmark_name = "nb301"
    yahpo_instances = ("CIFAR10",)
