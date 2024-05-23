from __future__ import annotations

from dataclasses import dataclass

from mfpbench.yahpo.benchmarks.iaml.iaml import IAMLBenchmark, IAMLConfig


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class IAMLrpartConfig(IAMLConfig):
    cp: float  # log
    maxdepth: int
    minbucket: int
    minsplit: int


class IAMLrpartBenchmark(IAMLBenchmark[IAMLrpartConfig]):
    yahpo_config_type = IAMLrpartConfig
    yahpo_has_conditionals = False
    yahpo_base_benchmark_name = "iaml_rpart"
    yahpo_instances = ("40981", "41146", "1489", "1067")
