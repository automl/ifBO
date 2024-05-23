from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.rbv2.rbv2 import RBV2Benchmark, RBV2Config


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class RBV2SuperConfig(RBV2Config):
    """Config has conditionals and as such, we use None to indicate not set."""

    learner_id: Literal["aknn", "glmnet", "ranger", "rpart", "svm", "xgboost"]
    num__impute__selected__cpo: Literal["impute.mean", "impute.median", "impute.hist"]

    # aknn
    aknn__M: int | None = None  # (18, 50)
    aknn__distance: Literal["l2", "cosine", "ip"] | None = None
    aknn__ef: int | None = None  # (7, 403),  log
    aknn__ef_construction: int | None = None  # (7, 1097),  log
    aknn__k: int | None = None  # (1, 50)

    glmnet__alpha: float | None = None  # (0.0, 1.0)
    glmnet__s: float | None = None  # (0.0009118819655545162, 1096.6331584284585), log

    ranger__min__node__size: int | None = None  # (1, 100)
    ranger__mtry__power: int | None = None  # (0, 1)
    ranger__num__random__splits: int | None = None  # (1, 100)
    ranger__num__trees: int | None = None  # (1, 2000)
    ranger__respect__unordered__factors: Literal[
        "ignore",
        "order",
        "partition",
    ] | None = None
    ranger__sample__fraction: float | None = None  # (0.1, 1.0)
    ranger__splitrule: Literal["gini", "extratrees"] | None = None

    rpart__cp: float | None = None  # (0.0009118819655545162, 1.0), log
    rpart__maxdepth: int | None = None  # (1, 30)
    rpart__minbucket: int | None = None  # (1, 100)
    rpart__minsplit: int | None = None  # (1, 100)

    svm__cost: float | None = None  # (4.5399929762484854e-05, 22026.465794806718), log
    svm__degree: int | None = None  # (2, 5)
    svm__gamma: float | None = None  # (4.5399929762484854e-05, 22026.465794806718), log
    svm__kernel: Literal["linear", "polynomial", "radial"] | None = None
    svm__tolerance: float | None = None  # (4.5399929762484854e-05, 2.0) log

    # (0.0009118819655545162, 1096.6331584284585), log
    xgboost__alpha: float | None = None
    xgboost__booster: Literal["gblinear", "gbtree", "dart"] | None = None
    xgboost__colsample_bylevel: float | None = None  # (0.01, 1.0)
    xgboost__colsample_bytree: float | None = None  # (0.01, 1.0)
    xgboost__eta: float | None = None  # (0.0009118819655545162, 1.0)  log
    # (4.5399929762484854e-05, 7.38905609893065), log
    xgboost__gamma: float | None = None
    # (0.0009118819655545162, 1096.6331584284585), log
    xgboost__lambda: float | None = None
    xgboost__max_depth: int | None = None  # (1, 15)
    # (2.718281828459045, 148.4131591025766),  log
    xgboost__min_child_weight: float | None = None
    xgboost__nrounds: int | None = None  # (7, 2981), log
    xgboost__rate_drop: float | None = None  # (0.0, 1.0)
    xgboost__skip_drop: float | None = None  # (0.0, 1.0)
    xgboost__subsample: float | None = None  # (0.1, 1.0)


class RBV2SuperBenchmark(RBV2Benchmark[RBV2SuperConfig]):
    yahpo_config_type = RBV2SuperConfig
    yapho_has_conditionals = True
    yahpo_base_benchmark_name = "rbv2_super"
    yahpo_instances = (
        "41138",
        "40981",
        "4134",
        "1220",
        "4154",
        "41163",
        "4538",
        "40978",
        "375",
        "1111",
        "40496",
        "40966",
        "4534",
        "40900",
        "40536",
        "41156",
        "1590",
        "1457",
        "458",
        "469",
        "41157",
        "11",
        "1461",
        "1462",
        "1464",
        "15",
        "40975",
        "41142",
        "40701",
        "40994",
        "23",
        "1468",
        "40668",
        "29",
        "31",
        "6332",
        "37",
        "40670",
        "23381",
        "151",
        "188",
        "41164",
        "1475",
        "1476",
        "1478",
        "41169",
        "1479",
        "41212",
        "1480",
        "300",
        "41143",
        "1053",
        "41027",
        "1067",
        "1063",
        "41162",
        "3",
        "6",
        "1485",
        "1056",
        "12",
        "14",
        "16",
        "18",
        "40979",
        "22",
        "1515",
        "334",
        "24",
        "1486",
        "1493",
        "28",
        "1487",
        "1068",
        "1050",
        "1049",
        "32",
        "1489",
        "470",
        "1494",
        "182",
        "312",
        "40984",
        "1501",
        "40685",
        "38",
        "42",
        "44",
        "46",
        "40982",
        "1040",
        "41146",
        "377",
        "40499",
        "50",
        "54",
        "307",
        "1497",
        "60",
        "1510",
        "40983",
        "40498",
        "181",
    )
