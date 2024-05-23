from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.iaml.iaml import IAMLBenchmark, IAMLConfig


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class IAMLxgboostConfig(IAMLConfig):
    nrounds: int  # log
    subsample: float
    alpha: float  # log
    _lambda: float  # log
    booster: Literal["gblinear", "gbtree", "dart"]

    colsample_bylevel: float | None = None
    colsample_bytree: float | None = None
    eta: float | None = None  # log
    gamma: float | None = None  # log
    max_depth: int | None = None
    min_child_weight: float | None = None  # log
    rate_drop: float | None = None
    skip_drop: float | None = None


class IAMLxgboostBenchmark(IAMLBenchmark[IAMLxgboostConfig]):
    _config_replacements: Mapping[str, str] = {"lambda": "_lambda"}
    yahpo_config_type = IAMLxgboostConfig
    yahpo_has_conditionals = True
    yahpo_base_benchmark_name = "iaml_xgboost"
    yahpo_instances = ("40981", "41146", "1489", "1067")
