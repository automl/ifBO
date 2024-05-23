from __future__ import annotations

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark, PD1ResultTransformer


class PD1lm1b_transformer_2048(PD1Benchmark):
    pd1_fidelity_range = (1, 74, 1)
    pd1_result_type = PD1ResultTransformer
    pd1_name = "lm1b-transformer-2048"

    @classmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "lr_decay_factor",
                    lower=0.010543,
                    upper=9.885653e-01,
                ),
                UniformFloatHyperparameter(
                    "lr_initial",
                    lower=0.000010,
                    upper=9.986256e00,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lr_power",
                    lower=0.100811,
                    upper=1.999659e00,
                ),
                UniformFloatHyperparameter(
                    "opt_momentum",
                    lower=0.000059,
                    upper=9.989986e-01,
                    log=True,
                ),
            ],
        )
        return cs
