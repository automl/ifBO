"""A ResultFrame is a mapping from a config to all results for that config."""
from __future__ import annotations

from itertools import chain
from typing import Any, Iterator, List, Mapping, Sequence, TypeVar, Union
from typing_extensions import Literal

import numpy as np

from mfpbench.config import Config
from mfpbench.result import Result
from mfpbench.stats import rank_correlation

C = TypeVar("C", bound=Config)
R = TypeVar("R", bound=Result)
F = TypeVar("F", int, float)

SENTINEL = object()


# This is a mapping from a
#   Config -> All results for the config over a list of fidelities
#   Fidelity -> All results for that fidelity
class ResultFrame(Mapping[Union[C, F], List[R]]):
    def __init__(self) -> None:  # noqa: D107
        # This lets us quickly index from a config to its registered results
        # We take this to be the rows and __len__
        self._ctor: dict[C, list[R]] = {}

        # This lets us quickly go from a fidelity to its registered results
        # This is akin to the columns
        self._ftor: dict[F, list[R]] = {}

        # This is an ordering for when a config result was added
        self._result_order: list[R] = []

    def __getitem__(self, key: C | F) -> list[R]:
        if isinstance(key, (int, float)):
            return self._ftor[key]

        if isinstance(key, Config):
            return self._ctor[key]

        raise KeyError(key)

    def __iter__(self) -> Iterator[C]:
        yield from iter(self._ctor)

    def __len__(self) -> int:
        return len(self._result_order)

    def add(self, result: R) -> None:
        """Add a result to the frame."""
        f = result.fidelity
        c = result.config

        if c in self._ctor:
            self._ctor[c].append(result)
        else:
            self._ctor[c] = [result]

        if f in self._ftor:
            self._ftor[f].append(result)
        else:
            self._ftor[f] = [result]

        self._result_order.append(result)

    def __contains__(self, key: C | F | Any) -> bool:
        if isinstance(key, (int, float)):
            return key in self._ftor
        if isinstance(key, Config):
            return key in self._ctor

        return False

    @property
    def fidelities(self) -> Iterator[F]:
        """Get the fidelities that have been evaluated."""
        yield from iter(self._ftor)

    @property
    def configs(self) -> Iterator[C]:
        """Get the configs that have been evaluated."""
        yield from iter(self._ctor)

    @property
    def results(self) -> Iterator[R]:
        """Get the results that have been evaluated."""
        yield from iter(self._result_order)

    def correlations(
        self,
        at: Sequence[F] | None = None,
        *,
        method: Literal["spearman", "kendalltau", "cosine"] = "spearman",
    ) -> np.ndarray:
        """The correlation ranksing between stored results.

        To calculate the correlations, we select all configs that are present in each
        selected fidelity.

        Args:
            at: The fidelities to get correlations between, defaults to all of them
            method: The method to calculate correlations with

        Returns:
            The correlation matrix with one row/column per fidelity
        """
        if len(self) == 0:
            raise RuntimeError("Must evaluate at two fidelities at least")
        if len(self._ftor) <= 1:
            raise ValueError(f"Only one fidelity {list(self._ftor)} evaluated")

        if at is None:
            at = list(self._ftor.keys())

        # Get the selected_fidelities
        # { 1: [a, b, c, ...], 2: [b, c, d, ...], ..., 100: [d, c, e, b, ...] }
        selected = {f: self._ftor[f] for f in at}

        # We get the intersection of configs that are found at all fidelity values
        # {b, c}
        common = {result.config for result in chain.from_iterable(selected.values())}

        # Next we prune out the selected fidelities results
        # {1: [b, c], 2: [b, c], ..., 100: [c, b]}
        # .. ensuring they're in some sorted order
        # {1: [b, c], 2: [b, c], ..., 100: [b, c]}
        selected = {
            f: sorted(
                [r for r in results if r.config in common],
                key=lambda r: repr(r.config),
            )
            for f, results in selected.items()
        }

        # Lastly, we pull out the results
        results = [
            [r.error for r in fidelity_results]
            for fidelity_results in selected.values()
        ]

        x = np.asarray(results)
        return rank_correlation(x, method=method)
