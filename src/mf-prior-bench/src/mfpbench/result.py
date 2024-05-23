from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Mapping, TypeVar
from typing_extensions import Self

from mfpbench.config import Config

if TYPE_CHECKING:
    from mfpbench.metric import Metric

# The Config kind
C = TypeVar("C", bound=Config)

# Fidelity kind
F = TypeVar("F", int, float)


@dataclass(frozen=True)  # type: ignore[misc]
class Result(ABC, Generic[C, F]):
    """Collect all results in a class for clarity."""

    metric_defs: ClassVar[Mapping[str, Metric]]
    """The metric definitions of this result."""

    default_value_metric: ClassVar[str]
    """The default metric to use for this result."""

    default_value_metric_test: ClassVar[str] | None
    """The default test metric to use for this result."""

    default_cost_metric: ClassVar[str]
    """The default cost to use for this result."""

    fidelity: F
    """The fidelity of this result."""

    config: C
    """The config used to generate this result."""

    value_metric: str
    """The metric to use for this result."""

    value_metric_test: str | None
    """The test metric to use for this result.

    !!! warning

        Some benchmarks do not have a test metric and so this may be
        None. You will likely want to check for this case if using the
        `value_metric_test`. A good approach is to replace with nan values.
    """

    cost_metric: str
    """The cost to use for this result."""

    @classmethod
    def from_dict(
        cls,
        config: C,
        fidelity: F,
        result: Mapping[str, float],
        *,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
        renames: Mapping[str, str] | None = None,
    ) -> Self:
        """Create from a dict or mapping object."""
        values = {
            k: (
                metric.as_value(v)
                if (metric := cls.metric_defs.get(k)) is not None
                else v
            )
            for k, v in result.items()
        }
        if renames is not None:
            values = {renames.get(k, k): v for k, v in values.items()}

        if value_metric is None:
            value_metric = cls.default_value_metric

        if value_metric_test is None:
            value_metric_test = cls.default_value_metric_test

        if cost_metric is None:
            cost_metric = cls.default_cost_metric

        return cls(
            config=config,
            fidelity=fidelity,
            value_metric=value_metric,
            value_metric_test=value_metric_test,
            cost_metric=cost_metric,
            **values,  # type: ignore
        )

    def as_dict(self) -> dict[str, Any]:
        """As a raw dictionary."""
        return self.values

    def __getitem__(self, key: str) -> Metric.Value:
        if key not in self.metric_defs:
            raise KeyError(f"Metric {key} not in {self.metric_defs.keys()}")
        return getattr(self, key)

    @property
    def cost(self) -> float:
        """The time cost for evaluting this config."""
        return self[self.cost_metric].error

    @property
    def error(self) -> float:
        """The error of interest."""
        return self[self.value_metric].error

    @property
    def test_error(self) -> float | None:
        """The test error of interest."""
        # TODO: This should just return None
        if self.value_metric_test is None:
            metric = self[self.value_metric].definition
            worst = metric.bounds[1] if metric.minimize else metric.bounds[0]
            return metric.as_value(worst).error

        return self[self.value_metric_test].error

    @property
    def score(self) -> float:
        """The score of interest."""
        return self[self.value_metric].score

    @property
    def val_score(self) -> float:
        """The score of interest."""
        # Maintain backwards compatibility for a project
        # TODO: This should be removed!
        return self.score

    @property
    def test_score(self) -> float | None:
        """The test score of interest."""
        # TODO: This should just return None
        if self.value_metric_test is None:
            metric = self[self.value_metric].definition
            worst = metric.bounds[1] if metric.minimize else metric.bounds[0]
            return metric.as_value(worst).score

        return self[self.value_metric_test].error

    @property
    def values(self) -> dict[str, Any]:
        """Create a dict from this result with the raw values."""
        return {k: getattr(self, k).value for k in self.metric_defs}

    @property
    def errors(self) -> dict[str, float]:
        """Create a dict from this result with the error values."""
        return {k: getattr(self, k).error for k in self.metric_defs}

    @property
    def scores(self) -> dict[str, float]:
        """Create a dict from this result with the score values."""
        return {k: getattr(self, k).score for k in self.metric_defs}
