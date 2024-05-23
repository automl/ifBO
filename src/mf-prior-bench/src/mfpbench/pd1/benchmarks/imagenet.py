from __future__ import annotations

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark, PD1ResultTransformer


class PD1imagenet_resnet_512(PD1Benchmark):
    pd1_result_type = PD1ResultTransformer
    pd1_fidelity_range = (3, 99, 1)
    pd1_name = "imagenet-resnet-512"

    @classmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "lr_decay_factor",
                    lower=0.010294,
                    upper=0.989753,
                ),
                UniformFloatHyperparameter(
                    "lr_initial",
                    lower=0.000010,
                    upper=9.774312,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lr_power",
                    lower=0.100225,
                    upper=1.999326,
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
