from __future__ import annotations

from copy import copy
from functools import reduce
from itertools import chain, tee
from typing import Callable, Iterable, Iterator, Mapping, TypeVar

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

T = TypeVar("T")


def findwhere(itr: Iterable[T], func: Callable[[T], bool], *, default: int = -1) -> int:
    """Find the index of the next occurence where func is True.

    Args:
        itr: The iterable to search over
        func: The function to use
        default: The default value to give if no value was found where func was True

    Returns:
        The first index where func was True
    """
    return next((i for i, t in enumerate(itr) if func(t)), default)


def remove_hyperparameter(name: str, space: ConfigurationSpace) -> ConfigurationSpace:
    """A new configuration space with the hyperparameter removed.

    Essentially copies hp over and fails if there is conditionals or forbiddens

    !!! warning

        This will not work with conditional search spaces and will raise an
        error

    Args:
        name: The name of the hyperparameter to remove
        space: The configuration space to remove the hyperparameter from

    Returns:
        The space with the hyperparameter removed
    """
    if name not in space._hyperparameters:
        raise ValueError(f"{name} not in {space}")

    if any(space.get_conditions()):
        raise NotImplementedError("We do not handle conditionals for now")

    if any(space.get_forbiddens()):
        raise NotImplementedError("We do not handle forbiddems for now")

    # Copying conditionals only work on objects and not named entities
    # Seeing as we copy objects and don't use the originals, transfering these
    # to the new objects is a bit tedious, possible but not required at this time
    # ... same goes for forbiddens
    assert name not in space._conditionals, "Can't handle conditionals"
    assert not any(
        name != f.hyperparameter.name for f in space.get_forbiddens()
    ), "Can't handle forbiddens"

    hps = [copy(hp) for hp in space.get_hyperparameters() if hp.name != name]

    if isinstance(space.random, np.random.RandomState):
        new_seed = space.random.randint(2**31 - 1)
    else:
        new_seed = copy(space.random)

    new_space = ConfigurationSpace(
        # TODO: not sure if this will have implications, assuming not
        seed=new_seed,
        name=copy(space.name),
        meta=copy(space.meta),
    )
    new_space.add_hyperparameters(hps)

    return new_space


def pairs(itr: Iterable[T]) -> Iterator[tuple[T, T]]:
    """An iterator over pairs of items in the iterator.

    ```python
    # Check if sorted
    if all(a < b for a, b in pairs(items)):
        ...
    ```

    Args:
        itr: An itr of items

    Returns:
        An itr of sequential pairs of the items
    """
    itr1, itr2 = tee(itr)

    # Skip first item
    _ = next(itr2)

    # Check there is a second element
    peek = next(itr2, None)
    if peek is None:
        raise ValueError("Can't create a pair from iterable with 1 item")

    # Put it back in
    itr2 = chain([peek], itr2)

    return iter((a, b) for a, b in zip(itr1, itr2))


def intersection(*items: Iterable[T]) -> set[T]:
    """Does an intersection over all collection of items.

    ```python
    ans = intersection(["a", "b", "c"], "ab", ("b", "c"))
    items = [(1, 2, 3), (2, 3), (4, 5)]
    ans = intesection(*items)
    ```

    Args:
        *items: Iterable things

    Returns:
        The intersection of all items
    """
    if len(items) == 0:
        return set()

    return set(reduce(lambda s1, s2: set(s1) & set(s2), items, items[0]))


K = TypeVar("K")
V = TypeVar("V")


def invert(d: Mapping[K, V]) -> Mapping[V, K]:
    """Invert the key value pairs of a dictionary."""
    return {v: k for k, v in d.items()}


K1 = TypeVar("K1")
K2 = TypeVar("K2")


def rename(d: Mapping[K1, V], keys: Mapping[K1, K2]) -> dict[K1 | K2, V]:
    """Rename keys of a dictionary based on a set of keys to update."""
    return {keys.get(k1, k1): v for k1, v in d.items()}


ValueT = TypeVar("ValueT", float, int, str)


def perturb(  # noqa: C901, PLR0912, PLR0911, PLR0915
    value: ValueT,
    hp: (
        Constant
        | UniformIntegerHyperparameter
        | UniformFloatHyperparameter
        | NormalIntegerHyperparameter
        | NormalFloatHyperparameter
        | CategoricalHyperparameter
        | OrdinalHyperparameter
    ),
    std: float,
    seed: int | np.random.RandomState | None = None,
) -> ValueT:
    """Perturb a value based on a hyperparameter.

    Args:
        value: The value to perturb
        hp: The hyperparameter it comes from
        std: The standard deviation of the noise to add
        seed: The seed to use for the perturbation

    Returns:
        The perturbed value
    """
    # TODO:
    # * https://github.com/automl/ConfigSpace/issues/289
    assert 0 <= std <= 1, "Noise must be between 0 and 1"
    rng: np.random.RandomState
    if seed is None:
        rng = np.random.RandomState()
    elif isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        rng = seed

    if isinstance(hp, Constant):
        return value

    if isinstance(
        hp,
        (
            NormalIntegerHyperparameter,
            NormalFloatHyperparameter,
            UniformFloatHyperparameter,
            UniformIntegerHyperparameter,
        ),
    ):
        # TODO:
        # * https://github.com/automl/ConfigSpace/issues/287
        # * https://github.com/automl/ConfigSpace/issues/290
        # * https://github.com/automl/ConfigSpace/issues/291
        # Doesn't act as intended
        assert hp.upper is not None
        assert hp.lower is not None
        assert hp.q is None
        assert isinstance(value, (int, float))

        if isinstance(hp, UniformIntegerHyperparameter):
            if hp.log:
                _lower = np.log(hp.lower)
                _upper = np.log(hp.upper)
            else:
                _lower = hp.lower
                _upper = hp.upper
        elif isinstance(hp, NormalIntegerHyperparameter):
            _lower = hp.nfhp._lower
            _upper = hp.nfhp._upper
        elif isinstance(hp, (UniformFloatHyperparameter, NormalFloatHyperparameter)):
            _lower = hp._lower
            _upper = hp._upper
        else:
            raise RuntimeError("Wut")

        space_length = std * (_upper - _lower)
        rescaled_std = std * space_length

        if not hp.log:
            sample = np.clip(rng.normal(value, rescaled_std), _lower, _upper)
        else:
            logged_value = np.log(value)
            sample = rng.normal(logged_value, rescaled_std)
            sample = np.clip(np.exp(sample), hp.lower, hp.upper)

        if isinstance(hp, (UniformIntegerHyperparameter, NormalIntegerHyperparameter)):
            return int(np.rint(sample))  # type: ignore

        if isinstance(hp, (UniformFloatHyperparameter, NormalFloatHyperparameter)):
            return float(sample)  # type: ignore

        raise RuntimeError("Please report to github, shouldn't get here")

        # if isinstance(hp, (BetaIntegerHyperparameter, BetaFloatHyperparameter)):
        # TODO
        # raise NotImplementedError(
        # "BetaIntegerHyperparameter, BetaFloatHyperparameter not implemented"
        # )

    if isinstance(hp, CategoricalHyperparameter):
        # We basically with (1 - std) choose the same value, otherwise uniformly select
        # at random
        if rng.uniform() < 1 - std:
            return value

        choices = set(hp.choices) - {value}
        return rng.choice(list(choices))  # type: ignore

    if isinstance(hp, OrdinalHyperparameter):
        # TODO:
        # * https://github.com/automl/ConfigSpace/issues/288
        # We build a normal centered at the value
        if rng.uniform() < 1 - std:
            return value

        # [0, 1,  2, 3]
        #       ^  mean
        index_value = hp.sequence.index(value)
        index_std = std * len(hp.sequence)
        normal_value = rng.normal(index_value, index_std)
        index = int(np.rint(np.clip(normal_value, 0, len(hp.sequence))))
        return hp.sequence[index]  # type: ignore

    raise ValueError(f"Can't perturb {hp}")
