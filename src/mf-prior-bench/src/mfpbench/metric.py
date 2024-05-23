from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class OutOfBoundsError(ValueError):
    """Raised when a value is outside of the bounds of a metric."""


@dataclass(frozen=True)
class Metric:
    """A metric to be used in the benchmark.

    It's main use is to convert a raw value into a value that can always be
    minimized.
    """

    minimize: bool
    """Whether or not to minimize the metric."""

    bounds: tuple[float, float] = field(default_factory=lambda: (-np.inf, np.inf))
    """The bounds of the metric."""

    def __post_init__(self) -> None:
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                f"bounds[0] must be less than bounds[1], got {self.bounds}",
            )

    def as_value(self, value: float) -> Metric.Value:
        """Convert a raw value into a metric value.

        Args:
            value: The raw value to convert.

        Returns:
            The metric value.
        """
        if pd.isna(value):
            value = np.inf
        return Metric.Value(value=value, definition=self)

    @property
    def optimum_value(self) -> Metric.Value:
        """Get the optimum value for this metric.

        Returns:
            The optimum value.
        """
        if self.minimize:
            return self.as_value(self.bounds[0])

        return self.as_value(self.bounds[1])

    @dataclass(frozen=True)
    class Value:
        """A value of a metric."""

        value: float
        definition: Metric = field(hash=False)

        def __post_init__(self) -> None:
            if not self.definition.bounds[0] <= self.value <= self.definition.bounds[1]:
                raise OutOfBoundsError(
                    f"Value {self.value} is outside of bounds {self.definition.bounds}",
                )

        @property
        def error(self) -> float:
            """Calculate a minimization value for the metric based on its raw value.

            The calculation is as follows:

                | direction | lower | upper |     | error                              |
                |-----------|-------|-------|-----|------------------------------------|
                | minimize  | inf   | inf   |     | value                              |
                | minimize  | A     | inf   |     | value                              |
                | minimize  | inf   | B     |     | value                              |
                | minimize  | A     | B     |     | abs(A - value) / abs(B - A)  # 0-1 |
                | ---       | ---   | ---   | --- | ---                                |
                | maximize  | inf   | inf   |     | -value                             |
                | maximize  | A     | inf   |     | -value                             |
                | maximize  | inf   | B     |     | -value                             |
                | maximize  | A     | B     |     | abs(B - value) / abs(B - a) # 0 -1 |

            Returns:
                The cost of the metric.
            """
            value = self.value
            lower, upper = self.definition.bounds
            if self.definition.minimize:
                if np.isinf(lower) or np.isinf(upper):
                    return value

                return abs(lower - value) / abs(upper - lower)

            if np.isinf(upper) or np.isinf(lower):
                return -value

            return abs(upper - value) / abs(upper - lower)

        @property
        def score(self) -> float:
            """Calculate a minimization value for the metric based on its raw value.

            The calculation is as follows:

                | direction | lower | upper |     | score                              |
                |-----------|-------|-------|-----|------------------------------------|
                | minimize  | inf   | inf   |     | -value                             |
                | minimize  | A     | inf   |     | -value                             |
                | minimize  | inf   | B     |     | -value                             |
                | minimize  | A     | B     |     | abs(B - value) / abs(B - A)  # 0-1 |
                | ---       | ---   | ---   | --- | ---                                |
                | maximize  | inf   | inf   |     | value                              |
                | maximize  | A     | inf   |     | value                              |
                | maximize  | inf   | B     |     | value                              |
                | maximize  | A     | B     |     | abs(A - value) / abs(B - A) # 0 -1 |

            Returns:
                The cost of the metric.
            """
            value = self.value
            lower, upper = self.definition.bounds
            if self.definition.minimize:
                if np.isinf(lower) or np.isinf(upper):
                    return -value

                return abs(upper - value) / abs(upper - lower)

            if np.isinf(upper) or np.isinf(lower):
                return value

            return abs(lower - value) / abs(upper - lower)
