from __future__ import annotations

from ConfigSpace import (
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

# This is used for n_estimators parameter of xgboost
MIN_ESTIMATORS = 50
MAX_ESTIMATORS = 2000


def space(seed: int | None) -> ConfigurationSpace:
    """Space for the xgboost surrogate."""
    cs = ConfigurationSpace(seed=seed)

    cs.add_hyperparameters(
        [
            UniformFloatHyperparameter(
                "learning_rate",
                lower=0.001,
                upper=1.0,
                default_value=0.3,
                log=True,
            ),  # learning rate
            UniformIntegerHyperparameter(
                "max_depth",
                lower=3,
                upper=20,
                default_value=10,
                log=True,
            ),
            UniformFloatHyperparameter(
                "colsample_bytree",
                lower=0.3,
                upper=1.0,
                default_value=1.0,
                log=True,
            ),
            UniformFloatHyperparameter(
                "reg_lambda",
                lower=1e-3,
                upper=10.0,
                default_value=1,
                log=True,
            ),
            UniformFloatHyperparameter(
                "subsample",
                lower=0.4,
                upper=1.0,
                default_value=1.0,
                log=False,
            ),
            UniformFloatHyperparameter(
                "alpha",
                lower=1e-3,
                upper=10,
                default_value=1,
                log=True,
            ),
            UniformIntegerHyperparameter(
                "min_child_weight",
                lower=1,
                upper=300,
                default_value=100,
            ),
        ],
    )
    return cs
