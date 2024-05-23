from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.iaml.iaml import IAMLBenchmark, IAMLConfig


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class IAMLSuperConfig(IAMLConfig):
    """Config has conditionals and as such, we use None to indicate not set."""

    learner_id: Literal["glmnet", "ranger", "rpart", "xgboost"]

    glmnet__alpha: float | None = None
    glmnet__s: float | None = None  # log

    ranger__min__node__size: int | None = None
    ranger__mtry__power: int | None = None
    ranger__num__random__splits: int | None = None
    ranger__num__trees: int | None = None
    ranger__respect__unordered__factors: Literal[
        "ignore",
        "order",
        "partition",
    ] | None = None
    ranger__sample__fraction: float | None = None
    ranger__splitrule: Literal["gini", "extratrees"] | None = None

    rpart__cp: float | None = None  # log
    rpart__maxdepth: int | None = None
    rpart__minbucket: int | None = None
    rpart__minsplit: int | None = None

    xgboost__alpha: float | None = None  # log
    xgboost__booster: Literal["gblinear", "gbtree", "dart"] | None = None
    xgboost__colsample_bylevel: float | None = None
    xgboost__colsample_bytree: float | None = None
    xgboost__eta: float | None = None  # log

    xgboost__gamma: float | None = None  # log
    xgboost__lambda: float | None = None  # log
    xgboost__max_depth: int | None = None
    xgboost__min_child_weight: float | None = None  # log
    xgboost__nrounds: int | None = None  # log
    xgboost__rate_drop: float | None = None
    xgboost__skip_drop: float | None = None
    xgboost__subsample: float | None = None


class IAMLSuperBenchmark(IAMLBenchmark[IAMLSuperConfig]):
    yahpo_config_type = IAMLSuperConfig
    yahpo_has_conditionals = True
    yahpo_base_benchmark_name = "iaml_super"
    yahpo_instances = ("40981", "41146", "1489", "1067")
