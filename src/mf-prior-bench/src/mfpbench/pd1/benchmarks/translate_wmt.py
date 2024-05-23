from __future__ import annotations

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark, PD1ResultTransformer


class PD1translatewmt_xformer_64(PD1Benchmark):
    pd1_fidelity_range = (1, 19, 1)
    pd1_result_type = PD1ResultTransformer
    pd1_name = "translate_wmt-xformer_translate-64"

    @classmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "lr_decay_factor",
                    lower=0.0100221257,
                    upper=0.988565263,
                ),
                UniformFloatHyperparameter(
                    "lr_initial",
                    lower=1.00276e-05,
                    upper=9.8422475735,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lr_power",
                    lower=0.1004250993,
                    upper=1.9985927056,
                ),
                UniformFloatHyperparameter(
                    "opt_momentum",
                    lower=5.86114e-05,
                    upper=0.9989999746,
                    log=True,
                ),
            ],
        )
        return cs
