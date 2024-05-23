"""The hartmann benchmarks.

The presets of terrible, bad, moderate and good are empirically obtained hyperparameters
for the hartmann function

The function flattens with increasing fidelity bias.
Along with increasing noise, that obviously makes one config harder to distinguish from
another.
Moreover, this works with any number of fidelitiy levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Mapping, TypeVar
from typing_extensions import override

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.benchmark import Benchmark
from mfpbench.config import Config
from mfpbench.metric import Metric
from mfpbench.result import Result
from mfpbench.synthetic.hartmann.generators import (
    MFHartmann3,
    MFHartmann6,
    MFHartmannGenerator,
)

G = TypeVar("G", bound=MFHartmannGenerator)


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class MFHartmann3Config(Config):
    X_0: float
    X_1: float
    X_2: float


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class MFHartmann6Config(Config):
    X_0: float
    X_1: float
    X_2: float
    X_3: float
    X_4: float
    X_5: float


@dataclass(frozen=True)  # type: ignore[misc]
class MFHartmann3Result(Result[MFHartmann3Config, int]):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        # TODO: There's probably some analytical upper bound...
        "value": Metric(minimize=True, bounds=(-3.86278, np.inf)),
        "fid_cost": Metric(minimize=True, bounds=(0.05, 1)),
    }
    default_value_metric: ClassVar[str] = "value"
    default_value_metric_test: ClassVar[None] = None
    default_cost_metric: ClassVar[str] = "fid_cost"

    value: Metric.Value
    fid_cost: Metric.Value


@dataclass(frozen=True)  # type: ignore[misc]
class MFHartmann6Result(Result[MFHartmann6Config, int]):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        # TODO: There's probably some analytical upper bound...
        "value": Metric(minimize=True, bounds=(-3.32237, np.inf)),
        "fid_cost": Metric(minimize=True, bounds=(0.05, 1)),
    }
    default_value_metric: ClassVar[str] = "value"
    default_value_metric_test: ClassVar[None] = None
    default_cost_metric: ClassVar[str] = "fid_cost"

    value: Metric.Value
    fid_cost: Metric.Value


C = TypeVar("C", bound=Config)
R = TypeVar("R", bound=Result)


class MFHartmannBenchmark(Benchmark[C, R, int], Generic[G, C, R]):
    mfh_generator_type: type[G]
    """The underlying mfhartmann function generator."""

    mfh_config_type: type[C]
    """The config type for this benchmark."""

    mfh_result_type: type[R]
    """The result type for this benchmark."""

    mfh_dims: ClassVar[int]
    """How many dimensions there are to the Hartmann function."""

    mfh_suffix: ClassVar[str]
    """Suffix for the benchmark name"""

    mfh_bias_noise: ClassVar[tuple[float, float]] = (0.5, 0.1)
    """The default bias and noise for mfhartmann benchmarks."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        bias: float | None = None,
        noise: float | None = None,
        prior: str | Path | C | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        cost_metric: str | None = None,
    ):
        """Initialize the benchmark.

        Args:
            seed: The seed to use.
            bias: How much bias to introduce
            noise: How much noise to introduce
            prior: The prior to use for the benchmark.

                * if `Path` - path to a file
                * if `Mapping` - Use directly
                * if `None` - There is no prior

            perturb_prior: If not None, will perturb the prior by this amount.
                For numericals, while for categoricals, this is interpreted as
                the probability of swapping the value for a random one.
            value_metric: The metric to use for this benchmark. Uses
                the default metric from the Result if None.
            cost_metric: The cost to use for this benchmark. Uses
                the default cost from the Result if None.
        """
        cls = self.__class__
        self.bias = bias if bias is not None else cls.mfh_bias_noise[0]
        self.noise = noise if noise is not None else cls.mfh_bias_noise[1]

        _max_fidelity = 100

        self.mfh = cls.mfh_generator_type(
            n_fidelities=_max_fidelity,
            fidelity_noise=self.noise,
            fidelity_bias=self.bias,
            seed=seed,
        )

        name = (
            f"mfh{cls.mfh_dims}_{cls.mfh_suffix}"
            if cls.mfh_suffix != ""
            else f"mfh{cls.mfh_dims}"
        )
        space = ConfigurationSpace(name=name, seed=seed)
        space.add_hyperparameters(
            [
                UniformFloatHyperparameter(f"X_{i}", lower=0.0, upper=1.0)
                for i in range(cls.mfh_dims)
            ],
        )
        super().__init__(
            name=name,
            config_type=self.mfh_config_type,
            result_type=self.mfh_result_type,
            fidelity_name="z",
            fidelity_range=(3, _max_fidelity, 1),
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
            value_metric=value_metric,
            cost_metric=cost_metric,
        )

    @override
    def _objective_function(
        self,
        config: Mapping[str, Any],
        *,
        at: int,
    ) -> dict[str, float]:
        query = dict(config)

        # It's important here that we still have X_0, X_1, ..., X_n
        # We strip out the numerical part and sort by that
        Xs = tuple(query[s] for s in sorted(query, key=lambda k: int(k.split("_")[-1])))
        return {"value": self.mfh(z=at, Xs=Xs), "fid_cost": self._fidelity_cost(at)}

    def _fidelity_cost(self, at: int) -> float:
        # λ(z) on Pg 18 from https://arxiv.org/pdf/1703.06240.pdf
        return 0.05 + (1 - 0.05) * (at / self.fidelity_range[1]) ** 2

    @property
    def optimum(self) -> C:
        """The optimum of the benchmark."""
        optimum = {f"X_{i}": x for i, x in enumerate(self.mfh_generator_type.optimum)}
        return self.Config.from_dict(optimum)


# -----------
# MFHartmann3
# -----------
class MFHartmann3Benchmark(
    MFHartmannBenchmark[
        MFHartmann3,
        MFHartmann3Config,
        MFHartmann3Result,
    ],
):
    mfh_generator_type = MFHartmann3
    mfh_config_type = MFHartmann3Config
    mfh_result_type = MFHartmann3Result
    mfh_dims = MFHartmann3.dims
    mfh_suffix = ""


class MFHartmann3BenchmarkTerrible(MFHartmann3Benchmark):
    mfh_bias_noise = (4.0, 5.0)
    mfh_suffix = "terrible"


class MFHartmann3BenchmarkBad(MFHartmann3Benchmark):
    mfh_bias_noise = (3.5, 4.0)
    mfh_suffix = "bad"


class MFHartmann3BenchmarkModerate(MFHartmann3Benchmark):
    mfh_bias_noise = (3.0, 3.0)
    mfh_suffix = "moderate"


class MFHartmann3BenchmarkGood(MFHartmann3Benchmark):
    mfh_bias_noise = (2.5, 2.0)
    mfh_suffix = "good"


# -----------
# MFHartmann6
# -----------
class MFHartmann6Benchmark(
    MFHartmannBenchmark[
        MFHartmann6,
        MFHartmann6Config,
        MFHartmann6Result,
    ],
):
    mfh_generator_type = MFHartmann6
    mfh_config_type = MFHartmann6Config
    mfh_result_type = MFHartmann6Result
    mfh_dims = MFHartmann6.dims
    mfh_suffix = ""


class MFHartmann6BenchmarkTerrible(MFHartmann6Benchmark):
    mfh_bias_noise = (4.0, 5.0)
    mfh_suffix = "terrible"


class MFHartmann6BenchmarkBad(MFHartmann6Benchmark):
    mfh_bias_noise = (3.5, 4.0)
    mfh_suffix = "bad"


class MFHartmann6BenchmarkModerate(MFHartmann6Benchmark):
    mfh_bias_noise = (3.0, 3.0)
    mfh_suffix = "moderate"


class MFHartmann6BenchmarkGood(MFHartmann6Benchmark):
    mfh_bias_noise = (2.5, 2.0)
    mfh_suffix = "good"
