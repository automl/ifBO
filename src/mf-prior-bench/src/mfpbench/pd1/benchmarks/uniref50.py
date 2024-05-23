from __future__ import annotations

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark, PD1ResultTransformer


class PD1uniref50_transformer_128(PD1Benchmark):
    pd1_fidelity_range = (1, 22, 1)
    pd1_result_type = PD1ResultTransformer
    pd1_name = "uniref50-transformer-128"

    @classmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "lr_decay_factor",
                    lower=0.0111588123,
                    upper=0.9898713967,
                ),
                UniformFloatHyperparameter(
                    "lr_initial",
                    lower=1.00564e-05,
                    upper=0.4429248972,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lr_power",
                    lower=0.1001570089,
                    upper=1.9989163336,
                ),
                UniformFloatHyperparameter(
                    "opt_momentum",
                    lower=5.86114e-05,
                    upper=0.9989940217,
                    log=True,
                ),
            ],
        )
        return cs
