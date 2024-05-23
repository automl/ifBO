from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Mapping, TypeVar
from typing_extensions import override

import numpy as np
import pandas as pd

from mfpbench.benchmark import Benchmark
from mfpbench.config import Config
from mfpbench.metric import Metric
from mfpbench.result import Result
from mfpbench.setup_benchmark import PD1Source

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from xgboost import XGBRegressor

PD1_FIDELITY_NAME = "epoch"


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class PD1Config(Config):
    """The config for PD1."""

    lr_decay_factor: float
    lr_initial: float
    lr_power: float
    opt_momentum: float


@dataclass(frozen=True)  # type: ignore[misc]
class PD1ResultSimple(Result[PD1Config, int]):
    """Used for all PD1 benchmarks, except imagenet, lm1b, translate_wmt, uniref50."""

    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "valid_error_rate": Metric(minimize=True, bounds=(0, 1)),
        "test_error_rate": Metric(minimize=True, bounds=(0, 1)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }
    default_value_metric: ClassVar[str] = "valid_error_rate"
    default_value_metric_test: ClassVar[str] = "test_error_rate"
    default_cost_metric: ClassVar[str] = "train_cost"

    valid_error_rate: Metric.Value
    test_error_rate: Metric.Value
    train_cost: Metric.Value


@dataclass(frozen=True)
class PD1ResultTransformer(Result[PD1Config, int]):
    """Imagenet, lm1b, translate_wmt, uniref50, cifar100 contains no test error."""

    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "valid_error_rate": Metric(minimize=True, bounds=(0, 1)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }
    default_value_metric: ClassVar[str] = "valid_error_rate"
    default_value_metric_test: ClassVar[None] = None
    default_cost_metric: ClassVar[str] = "train_cost"

    valid_error_rate: Metric.Value
    train_cost: Metric.Value


R = TypeVar("R", bound=Result)


class PD1Benchmark(Benchmark[PD1Config, R, int]):
    pd1_fidelity_range: ClassVar[tuple[int, int, int]]
    """The fidelity range for this benchmark."""

    pd1_name: ClassVar[str]
    """The name to access surrogates from."""

    pd1_result_type: type[R]
    """The result type for this benchmark."""

    def __init__(
        self,
        *,
        datadir: str | Path | None = None,
        seed: int | None = None,
        prior: str | Path | PD1Config | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        cost_metric: str | None = None,
    ):
        """Create a PD1 Benchmark.

        Args:
            datadir: Path to the data directory
            seed: The seed to use for the space
            prior: Any prior to use for the benchmark
            perturb_prior: Whether to perturb the prior. If specified, this
                is interpreted as the std of a normal from which to perturb
                numerical hyperparameters of the prior, and the raw probability
                of swapping a categorical value.
            value_metric: The metric to use for this benchmark. Uses
                the default metric from the Result if None.
            cost_metric: The cost to use for this benchmark. Uses
                the default cost from the Result if None.
        """
        cls = self.__class__
        space = cls._create_space(seed=seed)
        if datadir is None:
            datadir = PD1Source.default_location()

        datadir = Path(datadir) if isinstance(datadir, str) else datadir
        if not datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {datadir}."
                f"\n`python -m mfpbench download --status --data-dir {datadir.parent}`",
            )
        self._surrogates: dict[str, XGBRegressor] | None = None
        self.datadir = datadir

        super().__init__(
            seed=seed,
            name=self.pd1_name,
            config_type=PD1Config,
            fidelity_name=PD1_FIDELITY_NAME,
            fidelity_range=cls.pd1_fidelity_range,
            result_type=cls.pd1_result_type,
            prior=prior,
            perturb_prior=perturb_prior,
            space=space,
            value_metric=value_metric,
            cost_metric=cost_metric,
        )

    def load(self) -> None:
        """Load the benchmark."""
        _ = self.surrogates  # Call up the surrogate into memory

    @property
    def surrogates(self) -> dict[str, XGBRegressor]:
        """The surrogates for this benchmark, one per metric."""
        if self._surrogates is None:
            from xgboost import XGBRegressor

            self._surrogates = {}
            for metric, path in self.surrogate_paths.items():
                if not path.exists():
                    raise FileNotFoundError(
                        f"Can't find surrogate at {path}."
                        "\n`python -m mfpbench download --status --data-dir "
                        f" {self.datadir.parent}",
                    )
                model = XGBRegressor()
                model.load_model(path)
                self._surrogates[metric] = model

        return self._surrogates

    @property
    def surrogate_dir(self) -> Path:
        """The directory where the surrogates are stored."""
        return self.datadir / "surrogates"

    @property
    def surrogate_paths(self) -> dict[str, Path]:
        """The paths to the surrogates."""
        return {
            metric: self.surrogate_dir / f"{self.name}-{metric}.json"
            for metric in self.Result.metric_defs
        }

    @override
    def _objective_function(
        self,
        config: Mapping[str, Any],
        at: int,
    ) -> dict[str, float]:
        return self._results_for(config, fidelities=[at])[0]

    @override
    def _trajectory(
        self,
        config: Mapping[str, Any],
        *,
        frm: int,
        to: int,
        step: int,
    ) -> Iterable[tuple[int, Mapping[str, float]]]:
        fidelities = list(self.iter_fidelities(frm, to, step))
        return zip(fidelities, self._results_for(config, fidelities))

    def _results_for(
        self,
        config: Mapping[str, Any],
        fidelities: Iterable[int],
    ) -> list[dict[str, float]]:
        # Add the fidelities into the query and make a dataframe
        c = dict(config)
        queries = [{**c, self.fidelity_name: f} for f in fidelities]
        xs = pd.DataFrame(queries)

        # Predict the metric for everything in the dataframe
        features = xs.columns
        for metric, surrogate in self.surrogates.items():
            metric_min, metric_max = self.Result.metric_defs[metric].bounds
            # We clip as sometimes the surrogate produces negative values
            xs[metric] = surrogate.predict(xs[features]).clip(
                min=metric_min,
                max=metric_max,
            )

        metrics = list(self.surrogates.keys())
        return [dict(r[metrics]) for _, r in xs.iterrows()]

    @classmethod
    @abstractmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        ...
