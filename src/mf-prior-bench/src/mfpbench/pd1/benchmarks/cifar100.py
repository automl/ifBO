from __future__ import annotations

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark, PD1ResultTransformer


class PD1cifar100_wideresnet_2048(PD1Benchmark):
    pd1_fidelity_range = (45, 199, 1)
    pd1_result_type = PD1ResultTransformer
    pd1_name = "cifar100-wide_resnet-2048"

    @classmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "lr_decay_factor",
                    lower=0.010093,
                    upper=0.989012,
                ),
                UniformFloatHyperparameter(
                    "lr_initial",
                    lower=0.000010,
                    upper=9.779176,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lr_power",
                    lower=0.100708,
                    upper=1.999376,
                ),
                UniformFloatHyperparameter(
                    "opt_momentum",
                    lower=0.000059,
                    upper=0.998993,
                    log=True,
                ),
            ],
        )
        return cs
