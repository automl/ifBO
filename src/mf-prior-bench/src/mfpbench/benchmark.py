from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    overload,
)

import numpy as np

from mfpbench.config import Config
from mfpbench.result import Result
from mfpbench.resultframe import ResultFrame

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from mfpbench.metric import Metric

HERE = Path(__file__).parent.parent
PRIOR_DIR = HERE / "priors"

# The kind of Config to the benchmark
C = TypeVar("C", bound=Config)

# The return value from a config query
R = TypeVar("R", bound=Result)

# The kind of fidelity used in the benchmark
F = TypeVar("F", int, float)


class Benchmark(Generic[C, R, F], ABC):
    """Base class for a Benchmark."""

    _default_prior_dir: ClassVar[Path] = PRIOR_DIR
    """The default directory for priors"""

    _result_renames: ClassVar[Mapping[str, str] | None] = None
    """Any renaming to be done to raw result names before being passed
    to the `Result` type. This can be useful if for example, the benchmark returns
    a result named `valid-error-rate` but the `Result` type expects
    `valid_error_rate`, as you can't have `-` in a python identifier.
    """

    _config_renames: ClassVar[Mapping[str, str] | None] = None
    """Any renaming to be done to raw result names before being passed
    to the `Config` type. This can be useful if for example, the benchmark returns
    a result named `lambda` which is a reserved keyword in python but the `Config`
    type expects `_lambda` as the key.
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        space: ConfigurationSpace,
        config_type: type[C],
        result_type: type[R],
        fidelity_range: tuple[F, F, F],
        fidelity_name: str,
        *,
        has_conditionals: bool = False,
        seed: int | None = None,
        prior: str | Path | C | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
    ):
        """Initialize the benchmark.

        Args:
            name: The name of this benchmark
            space: The configuration space to use for the benchmark.
            config_type: The type of config to use for the benchmark.
            result_type: The type of result to use for the benchmark.
            fidelity_name: The name of the fidelity to use for the benchmark.
            fidelity_range: The range of fidelities to use for the benchmark.
            has_conditionals: Whether this benchmark has conditionals in it or not.
            seed: The seed to use.
            prior: The prior to use for the benchmark. If None, no prior is used.
                If a str, will check the local location first for a prior
                specific for this benchmark, otherwise assumes it to be a Path.
                If a Path, will load the prior from the path.
                If a Mapping, will be used directly.
            perturb_prior: If not None, will perturb the prior by this amount.
                For numericals, this is interpreted as the standard deviation of a
                normal distribution while for categoricals, this is interpreted
                as the probability of swapping the value for a random one.
            value_metric: The metric to use for this benchmark. Uses
                the default metric from the Result if None.
            value_metric_test: The metric to use as a test metric for this benchmark.
                Uses the default test metric from the Result if left as None, and
                if there is no default test metric, will return None.
            cost_metric: The cost to use for this benchmark. Uses
                the default cost from the Result if None.
        """
        if value_metric is None:
            value_metric = result_type.default_value_metric
        if value_metric_test is None:
            value_metric_test = result_type.default_value_metric_test

        if cost_metric is None:
            cost_metric = result_type.default_cost_metric

        # Ensure that the result type actually has an atrribute called value_metric
        if value_metric is None:
            assert getattr(self.Result, "value_metric", None) is not None
            value_metric = self.Result.value_metric

        self.name = name
        self.seed = seed
        self.space = space
        self.value_metric = value_metric
        self.value_metric_test: str | None = value_metric_test
        self.cost_metric = cost_metric
        self.fidelity_range: tuple[F, F, F] = fidelity_range
        self.fidelity_name = fidelity_name
        self.has_conditionals = has_conditionals
        self.Config = config_type
        self.Result = result_type
        self.metric_optimums = {
            metric_name: metric.optimum_value
            for metric_name, metric in self.Result.metric_defs.items()
        }

        self._prior_arg = prior

        # NOTE: This is handled entirely by subclasses as it requires knowledge
        # of the overall space the prior comes from, which only the subclasses now
        # at construction time. There's probably a better way to handle this but
        # for now this is fine.
        if perturb_prior is not None and not (0 <= perturb_prior <= 1):
            raise NotImplementedError(
                "If perturbing prior, `perturb_prior` must be in [0, 1]",
            )

        self.perturb_prior = perturb_prior
        self.prior: C | None = None

        if prior is not None:
            self.prior = self._load_prior(prior, benchname=self.name)
        else:
            self.prior = None

        if self.prior is not None and self.perturb_prior is not None:
            self.prior = self.prior.perturb(
                space,
                seed=self.seed,
                std=self.perturb_prior,
                categorical_swap_chance=self.perturb_prior,
            )

        if self.prior is not None:
            self.prior.set_as_default_prior(space)

    @property
    def metrics(self) -> dict[str, Metric]:
        """The metrics for this benchmark."""
        return dict(self.Result.metric_defs)

    @property
    def start(self) -> F:
        """The start of the fidelity range."""
        return self.fidelity_range[0]

    @property
    def end(self) -> F:
        """The end of the fidelity range."""
        return self.fidelity_range[1]

    @property
    def step(self) -> F:
        """The step of the fidelity range."""
        return self.fidelity_range[2]

    def _load_prior(
        self,
        prior: str | Path | Mapping[str, Any] | C,
        benchname: str | None = None,
    ) -> C:
        Config: type[C] = self.Config  # Need to be a bit explicit here

        if isinstance(prior, str):
            # It's a str, use as a key into available priors
            if benchname is not None:
                assumed_path = self._default_prior_dir / f"{benchname}-{prior}.yaml"
                if assumed_path.exists():
                    return Config.from_file(assumed_path)

            # Else we consider the prior to be a str reprsenting a Path
            return Config.from_file(Path(prior))

        if isinstance(prior, Path):
            return Config.from_file(prior)

        if isinstance(prior, Config):
            return prior

        if isinstance(prior, Mapping):
            return Config.from_dict(prior, renames=self._config_renames)

        raise ValueError(f"Unknown prior type {type(prior)}")

    def iter_fidelities(
        self,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
    ) -> Iterator[F]:
        """Iterate through the advertised fidelity space of the benchmark.

        Args:
            frm: Start of the curve, defaults to the minimum fidelity
            to: End of the curve, defaults to the maximum fidelity
            step: Step size, defaults to benchmark standard (1 for epoch)

        Returns:
            An iterator over the fidelities
        """
        frm = frm if frm is not None else self.start
        to = to if to is not None else self.end
        step = step if step is not None else self.step
        assert self.start <= frm <= to <= self.end

        dtype = int if isinstance(frm, int) else float
        fidelities: list[F] = list(
            np.arange(start=frm, stop=(to + step), step=step, dtype=dtype),
        )

        # Note: Clamping floats on arange
        #
        #   There's an annoying detail about floats here, essentially we could over
        #   (frm=0.03, to + step = 1+ .05, step=0.5) -> [0.03, 0.08, ..., 1.03]
        #   We just clamp this to the last fidelity
        #
        #   This does not effect ints
        if isinstance(step, float) and fidelities[-1] >= self.end:
            fidelities[-1] = self.end

        yield from fidelities

    def load(self) -> None:
        """Explicitly load the benchmark before querying, optional."""

    def query(
        self,
        config: C | Mapping[str, Any],
        *,
        at: F | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
    ) -> R:
        """Submit a query and get a result.

        Args:
            config: The query to use
            at: The fidelity at which to query, defaults to None which means *maximum*
            value_metric: The metric to use for this result. Uses
                the value metric passed in to the constructor if not specified,
                otherwise the default metric from the Result if None.
            value_metric: The metric to use for this result. Uses
                the value metric passed in to the constructor if not specified,
                otherwise the default metric from the Result if None.
            value_metric_test: The metric to use for this result. Uses
                the value metric passed in to the constructor if not specified,
                otherwise the default metric from the Result if None. If that
                is still None, then the `value_metric_test` will be None as well.
            cost_metric: The metric to use for this result. Uses
                the cost metric passed in to the constructor if not specified,
                otherwise the default metric from the Result if None.

        Returns:
            The result of the query
        """
        at = at if at is not None else self.end
        assert self.start <= at <= self.end

        if not isinstance(config, self.Config):
            _config = self.Config.from_dict(config, renames=self._config_renames)
        else:
            _config = config

        __config = dict(_config)
        if self._config_renames is not None:
            _reverse_renames = {v: k for k, v in self._config_renames.items()}
            __config = {k: __config.get(v, v) for k, v in _reverse_renames.items()}

        value_metric = value_metric if value_metric is not None else self.value_metric
        value_metric_test = (
            value_metric_test
            if value_metric_test is not None
            else self.value_metric_test
        )
        cost_metric = cost_metric if cost_metric is not None else self.cost_metric

        return self.Result.from_dict(
            config=config,
            fidelity=at,
            result=self._objective_function(__config, at=at),
            value_metric=str(value_metric),
            value_metric_test=value_metric_test,
            cost_metric=str(cost_metric),
            renames=self._result_renames,
        )

    def trajectory(
        self,
        config: C | Mapping[str, Any],
        *,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
    ) -> list[R]:
        """Get the full trajectory of a configuration.

        Args:
            config: The config to query
            frm: Start of the curve, should default to the start
            to: End of the curve, should default to the total
            step: Step size, defaults to ``cls.default_step``
            value_metric: The metric to use for this result. Uses
                the value metric passed in to the constructor if not specified,
                otherwise the default metric from the Result if None.
            value_metric_test: The metric to use for this result. Uses
                the value metric passed in to the constructor if not specified,
                otherwise the default metric from the Result if None. If that
                is still None, then the `value_metric_test` will be None as well.
            cost_metric: The metric to use for this result. Uses
                the cost metric passed in to the constructor if not specified,
                otherwise the default metric from the Result if None.

        Returns:
            A list of the results for this config
        """
        to = to if to is not None else self.end
        frm = frm if frm is not None else self.start
        step = step if step is not None else self.step

        __config = dict(config)
        if self._config_renames is not None:
            _reverse_renames = {v: k for k, v in self._config_renames.items()}
            __config = {k: __config.get(v, v) for k, v in _reverse_renames.items()}

        value_metric = value_metric if value_metric is not None else self.value_metric
        value_metric_test = (
            value_metric_test
            if value_metric_test is not None
            else self.value_metric_test
        )
        cost_metric = cost_metric if cost_metric is not None else self.cost_metric

        return [
            self.Result.from_dict(
                config=config,
                fidelity=fidelity,
                result=result,
                value_metric=str(value_metric),
                value_metric_test=value_metric_test,
                cost_metric=str(cost_metric),
                renames=self._result_renames,
            )
            for fidelity, result in self._trajectory(
                __config,
                frm=frm,
                to=to,
                step=step,
            )
        ]

    @abstractmethod
    def _objective_function(
        self,
        config: Mapping[str, Any],
        *,
        at: F,
    ) -> Mapping[str, float]:
        """Get the value of the benchmark for a config at a fidelity.

        Args:
            config: The config to query
            at: The fidelity to get the result at

        Returns:
            The result of the config as key value pairs
        """
        ...

    def _trajectory(
        self,
        config: Mapping[str, Any],
        *,
        frm: F,
        to: F,
        step: F,
    ) -> Iterable[tuple[F, Mapping[str, float]]]:
        """Get the trajectory of a config.

        By default this will just call the
        [`_objective_function()`][mfpbench.Benchmark._objective_function] for
        each fidelity but this can be overwritten if this can be done more optimaly.

        Args:
            config: The config to query
            frm: Start of the curve.
            to: End of the curve.
            step: Step size.

        Returns:
            A list of the results for this config
        """
        return [
            (fidelity, self._objective_function(config, at=fidelity))
            for fidelity in self.iter_fidelities(frm=frm, to=to, step=step)
        ]

    # No number specified, just return one config
    @overload
    def sample(
        self,
        n: None = None,
        *,
        seed: int | np.random.RandomState | None = None,
    ) -> C:
        ...

    # With a number, return many in a list
    @overload
    def sample(
        self,
        n: int,
        *,
        seed: int | np.random.RandomState | None = None,
    ) -> list[C]:
        ...

    def sample(
        self,
        n: int | None = None,
        *,
        seed: int | np.random.RandomState | None = None,
    ) -> C | list[C]:
        """Sample a random possible config.

        Args:
            n: How many samples to take, None means jsut a single one, not in a list
            seed: The seed to use for sampling

                !!! note "Seeding"

                    This is different than any seed passed to the construction
                    of the benchmark.

        Returns:
            Get back a possible Config to use
        """
        space = copy.deepcopy(self.space)
        if isinstance(seed, np.random.RandomState):
            rng = seed.randint(0, 2**31 - 1)
        else:
            rng = (
                seed
                if seed is not None
                else np.random.default_rng().integers(0, 2**31 - 1)
            )

        space.seed(rng)
        if n is None:
            return self.Config.from_dict(
                space.sample_configuration(),
                renames=self._config_renames,
            )

        # Just because of how configspace works
        if n == 1:
            return [
                self.Config.from_dict(
                    space.sample_configuration(),
                    renames=self._config_renames,
                ),
            ]

        return [
            self.Config.from_dict(c, renames=self._config_renames)
            for c in space.sample_configuration(n)
        ]

    def frame(self) -> ResultFrame[C, F, R]:
        """Get an empty frame to record with."""
        return ResultFrame[C, F, R]()
