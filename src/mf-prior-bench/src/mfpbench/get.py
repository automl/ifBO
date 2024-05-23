from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mfpbench.jahs import JAHSBenchmark
from mfpbench.lcbench_tabular import LCBenchTabularBenchmark
from mfpbench.nb201_tabular.benchmark import NB201TabularBenchmark
from mfpbench.pd1 import (
    PD1cifar100_wideresnet_2048,
    PD1imagenet_resnet_512,
    PD1lm1b_transformer_2048,
    PD1translatewmt_xformer_64,
    PD1uniref50_transformer_128,
)
from mfpbench.pd1_tabular import PD1TabularBenchmark
from mfpbench.synthetic.hartmann import (
    MFHartmann3Benchmark,
    MFHartmann3BenchmarkBad,
    MFHartmann3BenchmarkGood,
    MFHartmann3BenchmarkModerate,
    MFHartmann3BenchmarkTerrible,
    MFHartmann6Benchmark,
    MFHartmann6BenchmarkBad,
    MFHartmann6BenchmarkGood,
    MFHartmann6BenchmarkModerate,
    MFHartmann6BenchmarkTerrible,
)
from mfpbench.taskset_tabular import TaskSetTabularBenchmark
from mfpbench.yahpo import (
    IAMLglmnetBenchmark,
    IAMLrangerBenchmark,
    IAMLrpartBenchmark,
    IAMLSuperBenchmark,
    IAMLxgboostBenchmark,
    LCBenchBenchmark,
    NB301Benchmark,
    RBV2aknnBenchmark,
    RBV2glmnetBenchmark,
    RBV2rangerBenchmark,
    RBV2rpartBenchmark,
    RBV2SuperBenchmark,
    RBV2svmBenchmark,
    RBV2xgboostBenchmark,
)

if TYPE_CHECKING:
    from mfpbench.benchmark import Benchmark
    from mfpbench.config import Config

_mapping: dict[str, type[Benchmark]] = {
    # JAHS
    "jahs": JAHSBenchmark,
    # MFH
    "mfh3": MFHartmann3Benchmark,
    "mfh6": MFHartmann6Benchmark,
    "mfh3_terrible": MFHartmann3BenchmarkTerrible,
    "mfh3_bad": MFHartmann3BenchmarkBad,
    "mfh3_moderate": MFHartmann3BenchmarkModerate,
    "mfh3_good": MFHartmann3BenchmarkGood,
    "mfh6_terrible": MFHartmann6BenchmarkTerrible,
    "mfh6_bad": MFHartmann6BenchmarkBad,
    "mfh6_moderate": MFHartmann6BenchmarkModerate,
    "mfh6_good": MFHartmann6BenchmarkGood,
    # YAHPO
    "lcbench": LCBenchBenchmark,
    "nb301": NB301Benchmark,
    "rbv2_super": RBV2SuperBenchmark,
    "rbv2_aknn": RBV2aknnBenchmark,
    "rbv2_glmnet": RBV2glmnetBenchmark,
    "rbv2_ranger": RBV2rangerBenchmark,
    "rbv2_rpart": RBV2rpartBenchmark,
    "rbv2_svm": RBV2svmBenchmark,
    "rbv2_xgboost": RBV2xgboostBenchmark,
    "iaml_glmnet": IAMLglmnetBenchmark,
    "iaml_ranger": IAMLrangerBenchmark,
    "iaml_rpart": IAMLrpartBenchmark,
    "iaml_super": IAMLSuperBenchmark,
    "iaml_xgboost": IAMLxgboostBenchmark,
    # PD1
    "lm1b_transformer_2048": PD1lm1b_transformer_2048,
    "uniref50_transformer_128": PD1uniref50_transformer_128,
    "translatewmt_xformer_64": PD1translatewmt_xformer_64,
    "cifar100_wideresnet_2048": PD1cifar100_wideresnet_2048,
    "imagenet_resnet_512": PD1imagenet_resnet_512,
    # LCBenchTabular
    "lcbench_tabular": LCBenchTabularBenchmark,
    # PD1Tabular
    "pd1_tabular": PD1TabularBenchmark,
    # TaskSetTabular
    "taskset_tabular": TaskSetTabularBenchmark,
    # nb201 tabular
    "nb201_tabular": NB201TabularBenchmark,
}


def get(
    name: str,
    *,
    value_metric: str | None = None,
    cost_metric: str | None = None,
    prior: str | Path | Config | None = None,
    preload: bool = False,
    **kwargs: Any,
) -> Benchmark:
    """Get a benchmark.

    Args:
        name: The name of the benchmark
        value_metric: The value metric to use for the benchmark. If not specified,
            the default value metric is used.
        cost_metric: The cost metric to use for the benchmark. If not specified,
            the default cost metric is used.
        prior: The prior to use for the benchmark.
            * str -
                If it ends in {.json} or {.yaml, .yml}, it will convert it to a path and
                use it as if it is a path to a config. Otherwise, it is treated as preset
            * Path - path to a file
            * Config - A Config object
            * None - Use the default if available
        preload: Whether to preload the benchmark data in
        **kwargs: Extra arguments, optional or required for other benchmarks. Please
            look up the associated benchmarks.

    For the `#!python **kwargs`, please see the benchmarks listed below by `name=`

    ??? note "`#!python name='lcbench'` (YAHPO-GYM)"

        Possible `#!python task_id=`:

        ::: mfpbench.LCBenchBenchmark.yahpo_instances

        ::: mfpbench.LCBenchBenchmark.__init__
            options:
                show_source: false

    ??? note "`#!python name='lm1b_transformer_2048'` (PD1)"

        ::: mfpbench.PD1lm1b_transformer_2048.__init__
            options:
                show_source: false

    ??? note "`#!python name='uniref50_transformer_128'` (PD1)"

        ::: mfpbench.PD1uniref50_transformer_128.__init__
            options:
                show_source: false

    ??? note "`#!python name='cifar100_wideresnet_2048'` (PD1)"

        ::: mfpbench.PD1cifar100_wideresnet_2048.__init__
            options:
                show_source: false

    ??? note "`#!python name='imagenet_resnet_512'` (PD1)"

        ::: mfpbench.PD1imagenet_resnet_512.__init__
            options:
                show_source: false

    ??? note "`#!python name='jahs'`"

        Possible `#!python task_id=`:

        ::: mfpbench.JAHSBenchmark.task_ids

        ::: mfpbench.JAHSBenchmark.__init__
            options:
                show_source: false

    ??? note "`#!python name='mfh3'`"

        ::: mfpbench.MFHartmann3Benchmark.__init__
            options:
                show_source: false

    ??? note "`#!python name='mfh6'`"

        ::: mfpbench.MFHartmann6Benchmark.__init__
            options:
                show_source: false

    ??? note "`#!python name='lcbench_tabular'`"

        Possible `#!python task_id=`:

        ::: mfpbench.LCBenchTabularBenchmark.task_ids

        ::: mfpbench.LCBenchTabularBenchmark.__init__
            options:
                show_source: false


    """  # noqa: E501
    b = _mapping.get(name, None)
    bench: Benchmark

    if b is None:
        raise ValueError(f"{name} is not a benchmark in {list(_mapping.keys())}")

    if isinstance(prior, str) and any(
        prior.endswith(suffix) for suffix in [".json", ".yaml", ".yml"]
    ):
        prior = Path(prior)

    bench = b(prior=prior, cost_metric=cost_metric, value_metric=value_metric, **kwargs)

    if preload:
        bench.load()

    return bench


__all__ = ["get"]
