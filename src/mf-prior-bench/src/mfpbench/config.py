from __future__ import annotations

import json
from abc import ABC
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterator, Mapping
from typing_extensions import Self, override

import numpy as np
import yaml
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
)

from mfpbench.util import perturb


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class Config(ABC, Mapping[str, Any]):
    """A Config used to query a benchmark.

    * Include all hyperparams
    * Includes the fidelity
    * Configuration and generation agnostic
    * Immutable to prevent accidental changes during running, mutate with copy and
        provide arguments as required.
    * Easy equality between configs
    """

    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, Any],
        renames: Mapping[str, str] | None = None,
    ) -> Self:
        """Create from a dict or mapping object."""
        if renames is not None:
            d = {renames.get(k, k): v for k, v in d.items()}

        field_names = {f.name for f in fields(cls)}
        if not field_names.issuperset(d.keys()):
            raise ValueError(f"Dict keys {d.keys()} must be a subset of {field_names}")

        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})

    def as_dict(self) -> dict[str, Any]:
        """As a raw dictionary."""
        return asdict(self)

    def mutate(self, **kwargs: Any) -> Self:
        """Copy a config and mutate it if needed."""
        return replace(self, **kwargs)

    def copy(self) -> Self:
        """Copy this config and mutate it if needed."""
        return replace(self)

    def perturb(
        self,
        space: ConfigurationSpace,
        *,
        seed: int | np.random.RandomState | None = None,
        std: float | None = None,
        categorical_swap_chance: float | None = None,
    ) -> Self:
        """Perturb this config.

        Add gaussian noise to each hyperparameter. The mean is centered at
        the current config.

        Args:
            space: The space to perturb in
            seed: The seed to use for the perturbation
            std: A value in [0, 1] representing the fraction of the hyperparameter range
                to use as the std. If None, will use keep the current value
            categorical_swap_chance:
                The probability that a categorical hyperparameter will be changed
                If None, will use keep the current value

        Returns:
            The perturbed config
        """
        new_values: dict = {}
        for name, value in self.items():
            hp = space[name]
            if isinstance(hp, CategoricalHyperparameter) and categorical_swap_chance:
                new_value = perturb(value, hp, seed=seed, std=categorical_swap_chance)
            elif not isinstance(hp, CategoricalHyperparameter) and std:
                new_value = perturb(value, hp, seed=seed, std=std)
            else:
                new_value = value

            new_values[name] = new_value

        return self.mutate(**new_values)

    def __eq__(self, that: Any) -> bool:
        """Equality is defined in terms of their dictionary repr."""
        this = self.as_dict()
        if isinstance(that, dict):
            that = that.copy()
        elif isinstance(that, Configuration):
            that = dict(that)
        elif isinstance(that, self.__class__):
            that = that.as_dict()
        else:
            return False

        this = {
            k: np.round(v, 10) if isinstance(v, float) else v for k, v in this.items()
        }
        _that = {
            k: np.round(v, 10) if isinstance(v, float) else v for k, v in that.items()
        }
        return this == _that

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __len__(self) -> int:
        return len(self.as_dict())

    def __iter__(self) -> Iterator[str]:
        return self.as_dict().__iter__()

    def set_as_default_prior(self, configspace: ConfigurationSpace) -> None:
        """Apply this configuration as a prior on a configspace.

        Args:
            configspace: The space to apply this config to
        """
        # We convert to dict incase there's any special transformation that happen
        d = self.as_dict()
        for k, v in d.items():
            hp = configspace[k]
            # https://github.com/automl/ConfigSpace/issues/270
            if isinstance(hp, Constant):
                if hp.default_value != v:
                    raise ValueError(
                        f"Constant {k} must be set to the fixed value"
                        f" {hp.default_value}, not {v}",
                    )
                # No need to do anything here
            else:
                hp.default_value = hp.check_default(v)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Load a config from a supported file type.

        Note:
        ----
        Only supports yaml and json for now
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")

        if path.suffix == "json":
            return cls.from_json(path)
        if path.suffix in ["yaml", "yml"]:
            return cls.from_yaml(path)

        # It has no file suffix, just try both
        try:
            return cls.from_yaml(path)
        except yaml.error.YAMLError:
            pass

        try:
            return cls.from_json(path)
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Path {path} is not valid yaml or json")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a config from a yaml file."""
        path = Path(path)
        with path.open("r") as f:
            d = yaml.safe_load(f)
            return cls.from_dict(d)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """Load a config from a json file."""
        path = Path(path)
        with path.open("r") as f:
            d = json.load(f)
            return cls.from_dict(d)

    def save(self, path: str | Path, format: str | None = None) -> None:
        """Save the config.

        Args:
            path: Where to save to. Will infer json or yaml based on filename
            format: The format to save as. Will use file suffix if not provided
        """
        d = self.as_dict()
        path = Path(path)
        if format is None:
            if path.suffix == "json":
                format = "json"
            elif path.suffix in ["yaml", "yml"]:
                format = "yaml"
            else:
                format = "yaml"

        if format == "yaml":
            with path.open("w") as f:
                yaml.dump(d, f)
        elif format == "json":
            with path.open("w") as f:
                json.dump(d, f)
        else:
            raise ValueError(f"unkown format `format={format}`")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TabularConfig(Config):
    id: str | None = field(hash=False)
    """The id of this config.

    !!! warning

        While this is not required for a config, it is likely required to query
        into a tabular benchmark.

        Reasons for this not existing is when you have created this
        [`from_dict`][mfpbench.TabularConfig.from_dict] with a dict that does not have
        an id key.
    """

    @override
    def as_dict(self, *, with_id: bool = False) -> Any:
        """As a raw dictionary.


        Args:
            with_id: Whether to include the id key
        """
        d = {**super().as_dict()}
        if not with_id:
            d.pop("id")
        return d

    @classmethod
    @override
    def from_dict(
        cls,
        d: Mapping[str, Any],
        renames: Mapping[str, str] | None = None,
    ) -> Self:
        """Create from a dict or mapping object."""
        if renames is not None:
            d = {renames.get(k, k): v for k, v in d.items()}
        else:
            d = dict(d)
        d.setdefault("id", None)
        return cls(**d)

    @classmethod
    def names(cls) -> list[str]:
        """The names of entries in this config."""
        return [f.name for f in fields(cls) if f.name not in ("id",)]
