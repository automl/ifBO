from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.iaml.iaml import IAMLBenchmark, IAMLConfig


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class IAMLrangerConfig(IAMLConfig):
    min__node__size: int
    mtry__power: int
    num__trees: int
    respect__unordered__factors: Literal["ignore", "order", "partition"]
    sample__fraction: float
    splitrule: Literal["gini", "extratrees"]

    num__random__splits: int | None = None


class IAMLrangerBenchmark(IAMLBenchmark[IAMLrangerConfig]):
    yahpo_config_type = IAMLrangerConfig
    yahpo_has_conditionals = True
    yahpo_base_benchmark_name = "iaml_ranger"
    yahpo_instances = ("40981", "41146", "1489", "1067")
