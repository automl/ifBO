from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, TypeVar, overload
from typing_extensions import override

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from more_itertools import first_true

from mfpbench.benchmark import Benchmark
from mfpbench.config import TabularConfig
from mfpbench.result import Result

if TYPE_CHECKING:
    from mfpbench.metric import Metric


# The kind of Config to the **tabular** benchmark
CTabular = TypeVar("CTabular", bound=TabularConfig)

# The return value from a config query
R = TypeVar("R", bound=Result)

# The kind of fidelity used in the benchmark
F = TypeVar("F", int, float)


class TabularBenchmark(Benchmark[CTabular, R, F]):
    def __init__(  # noqa: PLR0913, C901
        self,
        name: str,
        table: pd.DataFrame,
        *,
        id_key: str,
        fidelity_key: str,
        result_type: type[R],
        config_type: type[CTabular],
        info_keys: Sequence[str] | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
        space: ConfigurationSpace | None = None,
        seed: int | None = None,
        prior: str | Path | CTabular | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
    ):
        """Initialize the benchmark.

        Args:
            name: The name of this benchmark.
            table: The table to use for the benchmark.
            id_key: The column in the table that contains the config id
            fidelity_key: The column in the table that contains the fidelity
            info_keys: Sequence of columns in the table that contain additional
                information. These will not be included in any config or results, and
                are only kept in the `.table` attribute.
            result_type: The result type for this benchmark.
            config_type: The config type for this benchmark.
            value_metric: The metric to use for this benchmark. Uses
                the default metric from the Result if None.
            value_metric_test: The metric to use as a test metric for this benchmark.
                Uses the default test metric from the Result if left as None, and
                if there is no default test metric, will return None.
            cost_metric: The cost to use for this benchmark. Uses
                the default cost from the Result if None.
            space: The configuration space to use for the benchmark. If None, will
                just be an empty space.
            prior: The prior to use for the benchmark. If None, no prior is used.
                If a string, will be treated as a prior specific for this benchmark
                if it can be found, otherwise assumes it to be a Path.
                If a Path, will load the prior from the path.
                If a dict or Configuration, will be used directly.
            perturb_prior: If not None, will perturb the prior by this amount.
                For numericals, while for categoricals, this is interpreted as the
                probability of swapping the value for a random one.
            seed: The seed to use for the benchmark.
        """
        # Make sure we work with a clean slate, no issue with index.
        table = table.reset_index()

        # Make sure all the keys they specified exist
        if id_key not in table.columns:
            raise ValueError(f"'{id_key=}' not in columns {table.columns}")

        if fidelity_key not in table.columns:
            raise ValueError(f"'{fidelity_key=}' not in columns {table.columns}")

        if info_keys is not None and not all(c in table.columns for c in info_keys):
            raise ValueError(f"'{info_keys=}' not in columns {table.columns}")

        result_keys: list[str] = list(result_type.metric_defs.keys())
        if not all(key in table.columns for key in result_keys):
            raise ValueError(
                f"Not all {result_keys=} not in columns {table.columns}",
            )

        config_keys: list[str] = config_type.names()
        if not all(key in table.columns for key in config_keys):
            raise ValueError(f"{config_keys=} not in columns {table.columns}")

        # Make sure that the column `id` only exist if it's the `id_key`
        if "id" in table.columns and id_key != "id":
            raise ValueError(
                f"Can't have `id` in the columns if it's not the {id_key=}."
                " Please drop it or rename it.",
            )

        # Remap their id column to `id`
        table = table.rename(columns={id_key: "id"}).astype({"id": str})

        # Index the table
        index_cols: list[str] = ["id", fidelity_key]

        # Drop all the columns that are not relevant
        relevant_cols: list[str] = [
            *index_cols,
            *result_keys,
            *config_keys,
        ]
        if info_keys is not None:
            relevant_cols.extend(info_keys)

        table = table[relevant_cols]  # type: ignore
        table = table.set_index(index_cols).sort_index()
        # MARK: put this back in after testing
        # table.index = table.index.set_levels(
        # [table.index.levels[0].astype(int), table.index.levels[1].astype(int)],
        # )

        # We now have the following table
        #
        #     id    fidelity | **metric, **config_values
        #     0         0    |
        #               1    |
        #               2    |
        #     1         0    |
        #               1    |
        #               2    |
        #   ...

        # Make sure we have equidistance fidelities for all configs
        fidelity_values = table.index.get_level_values(fidelity_key)
        # MARK: Comment this out after testing
        fidelity_counts = fidelity_values.value_counts()
        if not (fidelity_counts == fidelity_counts.iloc[0]).all():
            raise ValueError(f"{fidelity_key=} not uniform. \n{fidelity_counts}")

        sorted_fids = sorted(fidelity_values.unique())
        start = sorted_fids[0]
        end = sorted_fids[-1]
        step = sorted_fids[1] - sorted_fids[0]

        # Here we get all the unique configs
        #     id    fidelity | **metric, **config_values
        #     0         0    |
        #     1         0    |
        #   ...
        id_table = table.groupby(level="id").agg("first")
        configs = {
            str(config_id): config_type.from_dict(
                {
                    **row[config_keys].to_dict(),  # type: ignore
                    "id": str(config_id),
                },
            )
            for config_id, row in id_table.iterrows()
        }

        # Create the configuration space
        if space is None:
            space = ConfigurationSpace(name, seed=seed)

        self.table = table
        self.configs = configs
        self.id_key = id_key
        self.fidelity_key = fidelity_key
        self.config_keys = sorted(config_keys)
        self.result_keys = sorted(result_keys)

        super().__init__(
            name=name,
            seed=seed,
            config_type=config_type,
            result_type=result_type,
            fidelity_name=fidelity_key,
            fidelity_range=(start, end, step),
            space=space,
            prior=prior,
            perturb_prior=perturb_prior,
            value_metric=value_metric,
            value_metric_test=value_metric_test,
            cost_metric=cost_metric,
        )

        _raw_optimums = {
            (k, metric): (
                float(table[k].min()) if metric.minimize else float(table[k].max())
            )
            for k, metric in self.Result.metric_defs.items()
        }
        self.table_optimums: dict[str, Metric.Value] = {
            k: metric.as_value(v) for (k, metric), v in _raw_optimums.items()
        }

        if self.value_metric not in self.result_keys:
            raise ValueError(f"{self.value_metric=} not in {self.result_keys}")

        if self.cost_metric not in self.result_keys:
            raise ValueError(f"{self.cost_metric=} not in {self.result_keys}")

    def query(
        self,
        config: CTabular | Mapping[str, Any] | str,
        *,
        at: F | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
    ) -> R:
        """Submit a query and get a result.

        !!! warning "Passing a raw config"

            If a mapping is passed (and **not** a [`Config`][mfpbench.Config] object),
            we will attempt to look for `id` in the mapping, to know which config to
            lookup.

            If this fails, we will try match the config to one of the configs in
            the benchmark.

            Prefer to pass the [`Config`][mfpbench.Config] object directly if possible.

        ??? note "Override"

            This function overrides the default
            [`query()`][mfpbench.Benchmark.query] to allow for this
            config matching

        Args:
            config: The query to use
            at: The fidelity at which to query, defaults to None which means *maximum*
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
        _config = self._find_config(config)

        at = at if at is not None else self.end
        assert self.start <= at <= self.end

        __config = _config.as_dict(with_id=True)
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

    @override
    def trajectory(
        self,
        config: CTabular | Mapping[str, Any] | str,
        *,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
    ) -> list[R]:
        """Submit a query and get a result.

        !!! warning "Passing a raw config"

            If a mapping is passed (and **not** a [`Config`][mfpbench.Config] object),
            we will attempt to look for `id` in the mapping, to know which config to
            lookup.

            If this fails, we will try match the config to one of the configs in
            the benchmark.

            Prefer to pass the [`Config`][mfpbench.Config] object directly if possible.

        ??? note "Override"

            This function overrides the default
            [`trajectory()`][mfpbench.Benchmark.trajectory] to allow for this
            config matching

        Args:
            config: The query to use
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
            The result of the query
        """
        _config = self._find_config(config)

        to = to if to is not None else self.end
        frm = frm if frm is not None else self.start
        step = step if step is not None else self.step

        __config = _config.as_dict(with_id=True)
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

    def _find_config(
        self,
        config: CTabular | Mapping[str, Any] | str | int,
    ) -> CTabular:
        # It's an interger but likely meant to be string
        # We don't do any numeric based lookups
        if isinstance(config, int):
            config = str(config)

        # It's a key into the self.configs dict
        if isinstance(config, str):
            return self.configs[config]

        # If's a Config, that's fine
        if isinstance(config, self.Config):
            if config.id not in self.configs:
                raise ValueError(
                    f"Config {config.id} not in {self.configs.keys()}",
                )
            return config

        # At this point, we assume we're basically dealing with a dictionary
        assert isinstance(config, Mapping)

        # Not sure how that ended up there, but we can at least handle that
        if self.id_key in config:
            _real_config_id = str(config[self.id_key])
            return self.configs[_real_config_id]

        # Also ... not sure but anywho
        if "id" in config:
            _id = str(config["id"])
            return self.configs[_id]

        # Alright, nothing worked, here we try to match the actual hyperparameter
        # values to what we have in our known configs and attempt to get the
        # id that way
        match = first_true(
            self.configs.values(),
            pred=lambda c: c.as_dict(with_id=False) == config,  # type: ignore
            default=None,
        )
        if match is None:
            raise ValueError(
                f"Could not find config matching {config}. Please pass the"
                f" `Config` object or specify the `id` in the {type(config)}",
            )
        return match

    @override
    def _objective_function(
        self,
        config: Mapping[str, Any],
        *,
        at: F,
    ) -> Mapping[str, float]:
        """Submit a query and get a result.

        Args:
            config: The query to use
            at: The fidelity at which to query

        Returns:
            The result of the query
        """
        config = dict(config)
        _id = config.pop("id")
        row = self.table.loc[(_id, at)]
        row.name = _id
        _config = dict(row[self.config_keys])
        if config != _config:
            raise ValueError(
                f"Config queried with is not equal to the one in the table with {_id=}."
                f"\nconfig provided {config=}"
                f"\nconfig in table {_config=}",
            )

        return dict(row[self.result_keys])

    @override
    def _trajectory(
        self,
        config: Mapping[str, Any],
        *,
        frm: F,
        to: F,
        step: F,
    ) -> Iterable[tuple[F, Mapping[str, float]]]:
        config = dict(config)
        _id = config.pop("id")
        rows = self.table.loc[(_id, frm):(_id, to):step]  # type: ignore
        first_config = dict(rows.iloc[0][self.config_keys])

        if config != first_config:
            raise ValueError(
                f"Config queried with is not equal to the one in the table with {_id=}."
                f"\nconfig provided {config=}"
                f"\nconfig in table {first_config=}",
            )

        return [
            (fidelity, dict(row[self.result_keys]))
            for (_, fidelity), row in rows.iterrows()
        ]

    # No number specified, just return one config
    @overload
    def sample(
        self,
        n: None = None,
        *,
        seed: int | np.random.RandomState | None = ...,
    ) -> CTabular:
        ...

    # With a number, return many in a list
    @overload
    def sample(
        self,
        n: int,
        *,
        seed: int | np.random.RandomState | None = ...,
    ) -> list[CTabular]:
        ...

    @override
    def sample(
        self,
        n: int | None = None,
        *,
        seed: int | np.random.RandomState | None = None,
    ) -> CTabular | list[CTabular]:
        """Sample a random possible config.

        Args:
            n: How many samples to take, None means jsut a single one, not in a list
            seed: The seed to use for the sampling.

                !!! note "Seeding"

                    This is different than any seed passed to the construction
                    of the benchmark.

        Returns:
            Get back a possible Config to use
        """
        _seed: int | None
        if isinstance(seed, np.random.RandomState):
            _seed = seed.random_integers(0, 2**31 - 1)
        else:
            _seed = seed

        rng = np.random.default_rng(seed=_seed)

        config_items: list[CTabular] = list(self.configs.values())
        n_configs = len(config_items)
        sample_amount = n if n is not None else 1

        if sample_amount > n_configs:
            raise ValueError(
                f"Can't sample {sample_amount} configs from {n_configs} configs",
            )

        indices = rng.choice(n_configs, size=sample_amount, replace=False)
        if n is None:
            first_index: int = indices[0]
            return config_items[first_index]

        return [config_items[i] for i in indices]


if __name__ == "__main__":
    HERE = Path(__file__).parent
    path = HERE.parent.parent / "data" / "lcbench-tabular" / "adult.parquet"
    table = pd.read_parquet(path)
    from mfpbench.lcbench_tabular import LCBenchTabularConfig, LCBenchTabularResult

    benchmark = TabularBenchmark(
        "toy",
        table,
        id_key="id",
        fidelity_key="epoch",
        result_type=LCBenchTabularResult,
        config_type=LCBenchTabularConfig,
    )
    # benchmark = LCBenchTabular(task="adult")
    all_configs = benchmark.configs  # type: ignore
    config_ids = list(all_configs.keys())
    configs = list(all_configs.values())

    config = benchmark.sample(seed=1)
    config_id = config.id

    result = benchmark.query(config, at=1)
    argmin_score = benchmark.query(config, at=42)

    trajectory = benchmark.trajectory(config, frm=1, to=10)

    # lcbench = LCBenchTabular(task="adult")
    # All the same stuff as above
