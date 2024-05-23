from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Mapping, TypeVar
from typing_extensions import override

from mfpbench.benchmark import Benchmark, Config, Result
from mfpbench.setup_benchmark import YAHPOSource
from mfpbench.util import remove_hyperparameter

if TYPE_CHECKING:
    import onnxruntime
    import yahpo_gym

_YAHPO_LOADED = False


def _yahpo_create_session(
    *,
    benchmark: yahpo_gym.BenchmarkSet,
    multithread: bool = True,
) -> onnxruntime.InferenceSession:
    """Get a session for the Yahpo Benchmark.

    !!! note "Reason"

        In onnx 1.16.0, the `onnxruntime.InferenceSession` is explicitly required
        to provide the `providers=` argument.

    Code is taken from `yahpo_gym.benchmark.BenchmarkSet.set_session` and modified

    Args:
        benchmark:
            The benchmark to use for the session.
        multithread:
            Should the ONNX session be allowed to leverage multithreading capabilities?
            Initialized to `True` but on some HPC clusters it may be needed to set this
            to `False`, depending on your setup. Only relevant if no session is given.

    Returns:
        The session to use for the benchmark.
    """
    import onnxruntime

    model_path = benchmark._get_model_path()
    if not Path(model_path).is_file():
        raise Exception(f"ONNX file {model_path} not found!")

    options = onnxruntime.SessionOptions()
    if not multithread:
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1

    return onnxruntime.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _ensure_yahpo_config_set(datapath: Path) -> None:
    """Ensure that the yahpo config is set for this process.

    Will do nothing if it's already set up for this process.

    !!! note "Multiple Simultaneous YAHPO runs"

        When you have multiple runs of YAHPO at once, the will contend for file
        access to the default `settings_path="~/.config/yahpo_gym"`, a yaml file that
        defines where the 'data_path' is. This includes writes which can cause
        processes to crash.

        This `LocalConfiguration` that holds the `settings_path` is created upon import
        of `yahpo_gym` and treated as a Singleton for the process. However the
        reads/writes only happen upon running of a benchmark. Therefore we must disable
        the race condition of multiple processes all trying to access the default.

        There are two methods to address this:

        1. Setting the `YAHPO_LOCAL_CONFIG` environment variable before running the
            script.
        2. Modifying the Singleton before it's usage (this approach).

        Option 1. is likely the intended approach, however we wish to remove this burden
        from the user. Therefore we are taking approach 2.

        This is done by assinging each process a unique id and creating duplicate
        configs in a specially assigned temporary directory.

        The downside of this approach is that it will create junk files in the tmpdir
        which we can not automatically cleanup. These will be located in
        `"tmpdir/yahpo_gym_tmp_configs_delete_me_freely"`.

    Args:
        datapath: The path to the data directory.
    """
    if _YAHPO_LOADED:
        return

    pid = os.getpid()
    uuid_str = str(uuid.uuid4())
    unique_process_id = f"{uuid_str}_{pid}"

    tmpdir = tempfile.gettempdir()
    yahpo_dump_dir = Path(tmpdir) / "yahpo_gym_tmp_configs_delete_me_freely"
    yahpo_dump_dir.mkdir(exist_ok=True)

    config_for_this_process = yahpo_dump_dir / f"config_{unique_process_id}.yaml"

    import yahpo_gym

    yahpo_gym.local_config.settings_path = config_for_this_process
    yahpo_gym.local_config.init_config(data_path=str(datapath))
    return


# A Yahpo Benchmark is parametrized by a YAHPOConfig, YAHPOResult and fidelity
C = TypeVar("C", bound=Config)
R = TypeVar("R", bound=Result)
F = TypeVar("F", int, float)


class YAHPOBenchmark(Benchmark[C, R, F]):
    yahpo_base_benchmark_name: ClassVar[str]
    """Base name of the yahpo benchmark."""

    yahpo_config_type: type[C]
    """The config type for this benchmark."""

    yahpo_result_type: type[R]
    """The result type for this benchmark."""

    yahpo_fidelity_name: ClassVar[str]
    """The name of the fidelity for this benchmark."""

    yahpo_fidelity_range: tuple[F, F, F]
    """The fidelity range for this benchmark."""

    yahpo_has_conditionals: ClassVar[bool] = False
    """Whether this benchmark has conditionals."""

    yahpo_instances: ClassVar[tuple[str, ...] | None] = None
    """The instances available for this benchmark, if Any."""

    yahpo_task_id_name: ClassVar[str | None] = None
    """Name of hp used to indicate task."""

    yahpo_forced_remove_hps: ClassVar[Mapping[str, int | float | str] | None] = None
    """Any hyperparameters that should be forcefully deleted from the space
    but have default values filled in"""

    def __init__(  # noqa: C901, PLR0912
        self,
        task_id: str,
        *,
        datadir: str | Path | None = None,
        seed: int | None = None,
        prior: str | Path | C | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        session: onnxruntime.InferenceSession | None = None,
        value_metric: str | None = None,
        cost_metric: str | None = None,
    ):
        """Initialize a Yahpo Benchmark.

        Args:
            task_id: The task id to choose.
            seed: The seed to use
            datadir: The path to where mfpbench stores it data. If left to `None`,
                will use the `_default_download_dir = ./data/yahpo-gym-data`.
            seed: The seed for the benchmark instance
            prior: The prior to use for the benchmark. If None, no prior is used.
                If a str, will check the local location first for a prior
                specific for this benchmark, otherwise assumes it to be a Path.
                If a Path, will load the prior from the path.
                If a Mapping, will be used directly.
            perturb_prior: If given, will perturb the prior by this amount. Only used if
                `prior=` is given as a config.
            session: The onnxruntime session to use. If None, will create a new one.

                !!! warning "Not for faint hearted"

                    This is only a backdoor for onnx compatibility issues with YahpoGym.
                    You are advised not to use this unless you know what you are doing.
            value_metric: The metric to use for this benchmark. Uses
                the default metric from the Result if None.
            cost_metric: The cost to use for this benchmark. Uses
                the default cost from the Result if None.
        """
        # Validation
        cls = self.__class__

        # These errors are maintainers errors, not user errors
        if cls.yahpo_forced_remove_hps is not None and cls.yahpo_has_conditionals:
            raise NotImplementedError(
                "Error setting up a YAHPO Benchmark with conditionals",
                " and forced hps",
            )

        if cls.yahpo_task_id_name is not None and cls.yahpo_has_conditionals:
            raise NotImplementedError(
                f"{self.name} has conditionals, can't remove task_id from space",
            )

        instances = cls.yahpo_instances
        if task_id is None and instances is not None:
            raise ValueError(f"{cls} requires a task in {instances}")
        if task_id is not None and instances is None:
            raise ValueError(f"{cls} has no instances, you passed {task_id}")
        if task_id is not None and instances and task_id not in instances:
            raise ValueError(f"{cls} requires a task in {instances}")

        if datadir is None:
            datadir = YAHPOSource.default_location()
        elif isinstance(datadir, str):
            datadir = Path(datadir)

        datadir = Path(datadir) if isinstance(datadir, str) else datadir
        if not datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {datadir}, have you run\n"
                f"`python -m mfpbench download --status --data-dir {datadir.parent}`",
            )
        _ensure_yahpo_config_set(datadir)

        import yahpo_gym

        if session is None:
            dummy_bench = yahpo_gym.BenchmarkSet(
                cls.yahpo_base_benchmark_name,
                instance=task_id,
                multithread=False,
                # HACK: Used to fix onnxruntime session issue with 1.16.0 where
                # `providers` is required. By setting these options, we prevent
                # the benchmark from automatically creating a session.
                # We will manually do so and set it later.
                active_session=False,
                session=None,
            )
            session = _yahpo_create_session(benchmark=dummy_bench)

        bench = yahpo_gym.BenchmarkSet(
            cls.yahpo_base_benchmark_name,
            instance=task_id,
            multithread=False,
            session=session,
        )

        name = f"{cls.yahpo_base_benchmark_name}-{task_id}"

        # These can have one or two fidelities
        # NOTE: seed is allowed to be int | None
        space = bench.get_opt_space(
            drop_fidelity_params=True,
            seed=seed,  # type: ignore
        )

        if cls.yahpo_task_id_name is not None:
            space = remove_hyperparameter(cls.yahpo_task_id_name, space)

        if cls.yahpo_forced_remove_hps is not None:
            names = space.get_hyperparameter_names()
            for key in cls.yahpo_forced_remove_hps:
                if key in names:
                    space = remove_hyperparameter(key, space)

        self._bench = bench
        self.datadir = datadir
        self.task_id = task_id
        super().__init__(
            name=name,
            seed=seed,
            config_type=cls.yahpo_config_type,
            result_type=cls.yahpo_result_type,
            fidelity_name=cls.yahpo_fidelity_name,
            fidelity_range=cls.yahpo_fidelity_range,  # type: ignore
            has_conditionals=cls.yahpo_has_conditionals,
            space=space,
            prior=prior,
            perturb_prior=perturb_prior,
            value_metric=value_metric,
            cost_metric=cost_metric,
        )

    @property
    def bench(self) -> yahpo_gym.BenchmarkSet:
        """The underlying yahpo gym benchmark."""
        if self._bench is None:
            import yahpo_gym

            bench = yahpo_gym.BenchmarkSet(
                self.yahpo_base_benchmark_name,
                instance=self.task_id,
                multithread=False,
            )
            self._bench = bench
        return self._bench

    def load(self) -> None:
        """Load the benchmark into memory."""
        _ = self.bench

    @override
    def _trajectory(
        self,
        config: Mapping[str, Any],
        *,
        frm: F,
        to: F,
        step: F,
    ) -> Iterable[tuple[F, Mapping[str, float]]]:
        query = dict(config)

        if self.yahpo_forced_remove_hps is not None:
            query.update(self.yahpo_forced_remove_hps)

        if self.task_id is not None and self.yahpo_task_id_name is not None:
            query[self.yahpo_task_id_name] = self.task_id

        # Copy same config and insert fidelities for each
        queries: list[dict] = [
            {**query, self.fidelity_name: f}
            for f in self.iter_fidelities(frm=frm, to=to, step=step)
        ]

        # NOTE: seed is allowed to be int | None
        results: list[dict] = self.bench.objective_function(
            queries,
            seed=self.seed,  # type: ignore
        )
        return zip(self.iter_fidelities(frm=frm, to=to, step=step), results)

    @override
    def _objective_function(self, config: Mapping[str, Any], at: F) -> dict[str, float]:
        query = dict(config)

        if self.yahpo_forced_remove_hps is not None:
            query.update(self.yahpo_forced_remove_hps)

        if self.task_id is not None and self.yahpo_task_id_name is not None:
            query[self.yahpo_task_id_name] = self.task_id

        query[self.fidelity_name] = at

        # NOTE: seed is allowed to be int | None
        results: list[dict] = self.bench.objective_function(
            query,
            seed=self.seed,  # type: ignore
        )
        return results[0]
