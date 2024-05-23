from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Literal, Mapping
from typing_extensions import override

import numpy as np
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
)

from mfpbench.benchmark import Benchmark
from mfpbench.config import Config
from mfpbench.metric import Metric
from mfpbench.result import Result
from mfpbench.setup_benchmark import JAHSBenchSource

if TYPE_CHECKING:
    import jahs_bench


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class JAHSConfig(Config):
    """The config for JAHSBench, useful to have regardless of the configspace used.

    https://github.com/automl/jahs_bench_201/blob/main/jahs_bench/lib/core/configspace.py
    """

    # Not fidelities for our use case
    N: int
    W: int

    # Categoricals
    Op1: int
    Op2: int
    Op3: int
    Op4: int
    Op5: int
    Op6: int
    TrivialAugment: bool
    Activation: str
    Optimizer: str

    # Continuous Numericals
    Resolution: float
    LearningRate: float
    WeightDecay: float


@dataclass(frozen=True)  # type: ignore[misc]
class JAHSResult(Result[JAHSConfig, int]):
    default_value_metric: ClassVar[str] = "valid_acc"
    default_cost_metric: ClassVar[str] = "runtime"
    default_value_metric_test: ClassVar[str] = "test_acc"

    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "runtime": Metric(minimize=True, bounds=(0, np.inf)),
        "valid_acc": Metric(minimize=False, bounds=(0, 100)),
        "test_acc": Metric(minimize=False, bounds=(0, 100)),
    }

    # Info
    # size: float  # remove
    # flops: float # remove
    # latency: float  # unit? remove
    runtime: Metric.Value  # unit?

    # Scores (0 - 100)
    valid_acc: Metric.Value
    test_acc: Metric.Value
    # train_acc: float # remove


class JAHSBenchmark(Benchmark[JAHSConfig, JAHSResult, int], ABC):
    JAHS_FIDELITY_NAME: ClassVar[str] = "epoch"
    JAHS_FIDELITY_RANGE: ClassVar[tuple[int, int, int]] = (3, 200, 1)
    JAHS_METRICS_TO_ACTIVATE: ClassVar[tuple[str, ...]] = (
        "valid-acc",
        "test-acc",
        "runtime",
    )

    task_ids: ClassVar[tuple[str, str, str]] = (
        "CIFAR10",
        "ColorectalHistology",
        "FashionMNIST",
    )
    """
    ```python exec="true" result="python"
    from mfpbench import JAHSBenchmark
    print(JAHSBenchmark.task_ids)
    ```
    """

    _result_renames: ClassVar[Mapping[str, str]] = {
        "size_MB": "size",
        "FLOPS": "flops",
        "valid-acc": "valid_acc",
        "test-acc": "test_acc",
        "train-acc": "train_acc",
    }

    def __init__(
        self,
        task_id: Literal["CIFAR10", "ColorectalHistology", "FashionMNIST"],
        *,
        datadir: str | Path | None = None,
        seed: int | None = None,
        prior: str | Path | JAHSConfig | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        cost_metric: str | None = None,
    ):
        """Initialize the benchmark.

        Args:
            task_id: The specific task to use.
            datadir: The path to where mfpbench stores it data. If left to `None`,
                will use `#!python _default_download_dir = "./data/jahs-bench-data"`.
            seed: The seed to give this benchmark instance
            prior: The prior to use for the benchmark.

                * if `str` - A preset
                * if `Path` - path to a file
                * if `dict`, Config, Configuration - A config
                * if `None` - Use the default if available

            perturb_prior: If given, will perturb the prior by this amount.
                Only used if `prior=` is given as a config.
            value_metric: The metric to use for this benchmark. Uses
                the default metric from the Result if None.
            cost_metric: The cost to use for this benchmark. Uses
                the default cost from the Result if None.
        """
        cls = self.__class__
        if datadir is None:
            datadir = JAHSBenchSource.default_location()

        datadir = Path(datadir)

        if not datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {datadir}."
                f"\n`python -m mfpbench download --status --data-dir {datadir}`",
            )

        # Loaded on demand with `@property`
        self._bench: jahs_bench.Benchmark | None = None
        self.datadir = datadir
        self.task_id = task_id

        name = f"jahs_{task_id}"
        super().__init__(
            seed=seed,
            name=name,
            config_type=JAHSConfig,
            result_type=JAHSResult,
            fidelity_name=self.JAHS_FIDELITY_NAME,
            fidelity_range=self.JAHS_FIDELITY_RANGE,
            space=cls._jahs_configspace(name=name, seed=seed),
            prior=prior,
            perturb_prior=perturb_prior,
            value_metric=value_metric,
            cost_metric=cost_metric,
        )

    # explicit overwrite
    def load(self) -> None:
        """Pre-load JAHS XGBoost model before querying the first time."""
        # Access the property
        _ = self.bench

    @property
    def bench(self) -> jahs_bench.Benchmark:
        """The underlying benchmark used."""
        if not self._bench:
            try:
                import jahs_bench
            except ImportError as e:
                raise ImportError(
                    "jahs-bench not installed, please install it with "
                    "`pip install jahs-bench`",
                ) from e

            tasks = {
                "CIFAR10": jahs_bench.BenchmarkTasks.CIFAR10,
                "ColorectalHistology": jahs_bench.BenchmarkTasks.ColorectalHistology,
                "FashionMNIST": jahs_bench.BenchmarkTasks.FashionMNIST,
            }
            task = tasks.get(self.task_id, None)
            if task is None:
                raise ValueError(
                    f"Unknown task {self.task_id}, must be in {list(tasks.keys())}",
                )

            self._bench = jahs_bench.Benchmark(
                task=self.task_id,
                save_dir=self.datadir,
                download=False,
                metrics=self.JAHS_METRICS_TO_ACTIVATE,
            )

        return self._bench

    @override
    def _objective_function(
        self,
        config: Mapping[str, Any],
        at: int,
    ) -> dict[str, float]:
        query = dict(config)
        results = self.bench.__call__(query, nepochs=at)
        return results[at]

    @override
    def _trajectory(
        self,
        config: Mapping[str, Any],
        *,
        frm: int,
        to: int,
        step: int,
    ) -> Iterable[tuple[int, Mapping[str, float]]]:
        query = dict(config)

        try:
            return self.bench.__call__(query, nepochs=to, full_trajectory=True).items()
        except TypeError:
            # See: https://github.com/automl/jahs_bench_201/issues/5
            # Revert back to calling individually, default behaviour
            return super()._trajectory(config, frm=frm, to=to, step=step)

    @classmethod
    def _jahs_configspace(
        cls,
        name: str = "jahs_bench_config_space",
        seed: int | np.random.RandomState | None = None,
    ) -> ConfigurationSpace:
        """The configuration space for all datasets in JAHSBench.

        Args:
            name: The name to give to the config space.
            seed: The seed to use for the config space

        Returns:
            The space
        """
        # Copied from https://github.com/automl/jahs_bench_201/blob/c1e92dd92a0c4906575c4e3e4ee9e7420efca5f1/jahs_bench/lib/core/configspace.py#L4  # noqa: E501
        # See for why we copy: https://github.com/automl/jahs_bench_201/issues/4
        if isinstance(seed, np.random.RandomState):
            seed = seed.tomaxint()

        try:
            from jahs_bench.lib.core.constants import Activations
        except ImportError as e:
            raise ImportError(
                "jahs-bench not installed, please install it with "
                "`pip install jahs-bench`",
            ) from e

        space = ConfigurationSpace(name=name, seed=seed)
        space.add_hyperparameters(
            [
                Constant(
                    "N",
                    # sequence=[1, 3, 5],
                    value=5,  # This is the value for NB201
                ),
                Constant(
                    "W",
                    # sequence=[4, 8, 16],
                    value=16,  # This is the value for NB201
                ),
                CategoricalHyperparameter(
                    "Op1",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op2",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op3",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op4",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op5",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op6",
                    choices=list(range(5)),
                    default_value=0,
                ),
                # OrdinalHyperparameter(
                #     "Resolution",
                #     sequence=[0.25, 0.5, 1.0],
                #     default_value=1.0,
                # ),
                Constant("Resolution", value=1.0),
                CategoricalHyperparameter(
                    "TrivialAugment",
                    choices=[True, False],
                    default_value=False,
                ),
                CategoricalHyperparameter(
                    "Activation",
                    choices=list(Activations.__members__.keys()),
                    default_value="ReLU",
                ),
            ],
        )

        # Add Optimizer related HyperParamters
        optimizers = Constant("Optimizer", value="SGD")
        lr = UniformFloatHyperparameter(
            "LearningRate",
            lower=1e-3,
            upper=1e0,
            default_value=1e-1,
            log=True,
        )
        weight_decay = UniformFloatHyperparameter(
            "WeightDecay",
            lower=1e-5,
            upper=1e-2,
            default_value=5e-4,
            log=True,
        )

        space.add_hyperparameters([optimizers, lr, weight_decay])
        return space
