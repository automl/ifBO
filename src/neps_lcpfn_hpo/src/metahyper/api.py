from __future__ import annotations

import os
import inspect
import logging
import shutil
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
import zipfile

from ._locker import Locker
from .utils import YamlSerializer, find_files, non_empty_file

warnings.simplefilter("always", DeprecationWarning)


@dataclass
class ConfigResult:
    config: Any
    result: dict
    metadata: dict

    def __lt__(self, other):
        return self.result["loss"] < other.result["loss"]


class ConfigInRun:
    config: Any | None = None
    config_id: str | None = None
    pipeline_directory: Path | str | None = None
    previous_pipeline_directory: Path | str | None = None
    optimization_dir: Path | str | None = None

    @staticmethod
    def store_in_run_data(
        config,
        config_id,
        pipeline_directory,
        previous_pipeline_directory,
        optimization_dir,
    ):
        ConfigInRun.config = config
        ConfigInRun.config_id = config_id
        ConfigInRun.pipeline_directory = Path(pipeline_directory)
        ConfigInRun.previous_pipeline_directory = previous_pipeline_directory
        ConfigInRun.optimization_dir = Path(optimization_dir)


class Sampler(ABC):
    # pylint: disable=no-self-use,unused-argument
    def __init__(self, budget: int | float | None = None):
        self.used_budget: int | float = 0
        self.budget = budget

    def get_state(self) -> Any:
        """Return a state for the sampler that will be used in every other thread"""
        state = {
            "used_budget": self.used_budget,
        }
        if self.budget is not None:
            state["remaining_budget"] = self.budget - self.used_budget
        return state

    def load_state(self, state: dict[str, Any]):
        """Load a state for the sampler shared accross threads"""
        self.used_budget = state["used_budget"]

    @contextmanager
    def using_state(self, state_file: Path, serializer):
        if state_file.exists():
            self.load_state(serializer.load(state_file))
        yield self

        serializer.dump(self.get_state(), state_file)

    def load_results(
        self, results: dict[Any, ConfigResult], pending_configs: dict[Any, ConfigResult]
    ) -> None:
        return

    @abstractmethod
    def get_config_and_ids(self) -> tuple[Any, str, str | None]:
        """Sample a new configuration

        Returns:
            config: serializable object representing the configuration
            config_id: unique identifier for the configuration
            previous_config_id: if provided, id of a previous on which this
                configuration is based
        """
        raise NotImplementedError

    def load_config(self, config: Any):
        """Transform a serialized object into a configuration object"""
        return config


class Configuration:
    """If the configuration is not a simple dictionary, it should inherit from
    this object and define the 'hp_values' method"""

    def hp_values(self):
        """Should return a dictionary of the hyperparameter values"""
        raise NotImplementedError


def _process_sampler_info(
    serializer: YamlSerializer,
    sampler_info: dict,
    sampler_info_file: Path,
    decision_locker: Locker,
    logger=None,
):
    """
    This function is called by the metahyper before sampling and training happens.
    It performs checks on the optimizer's YAML data file to ensure data integrity
    and applies sanity checks for potential user errors when running NePS.

    The function utilizes a file-locking mechanism using the `Locker` class to ensure
    thread safety, preventing potential conflicts when multiple threads access the file
    simultaneously.

    Parameters:
        - serializer: The YAML serializer object used for loading and dumping data.
        - sampler_info: The dictionary containing information for the optimizer.
        - sampler_info_file: The path to the YAML file storing optimizer data if available.
        - decision_locker: The Locker file to use during multi-thread communication.
        - logger: An optional logger object for logging messages (default is 'neps').

    Note:
        - The file-locking mechanism is employed to avoid potential errors in multiple threads.
        - The `Locker` class and `YamlSerializer` should be appropriately defined or imported.
        - Ensure that potential side effects or dependencies are considered when using this function.
    """
    if logger is None:
        logger = logging.getLogger("neps")

    # should_break = False
    # while not should_break:
        # if decision_locker.acquire_lock():
    try:
        if sampler_info_file.exists():
            optimizer_data = serializer.load(sampler_info_file)
            excluded_key = "searcher_name"
            sampler_info_copy = sampler_info.copy()
            optimizer_data_copy = optimizer_data.copy()
            sampler_info_copy.pop(excluded_key, None)
            optimizer_data_copy.pop(excluded_key, None)

            if sampler_info_copy != optimizer_data_copy:
                raise ValueError(
                    f"The sampler_info in the file {sampler_info_file} is not valid. "
                    f"Expected: {sampler_info_copy}, Found: {optimizer_data_copy}"
                )
        else:
            # If the file is empty or doesn't exist, write the sampler_info
            serializer.dump(sampler_info, sampler_info_file, sort_keys=False)
    except Exception as e:
        raise RuntimeError(f"Error during data saving: {e}") from e
    # finally:
        #     decision_locker.release_lock()
        #     should_break = True


def _load_sampled_paths(optimization_dir: Path | str, serializer, logger):
    optimization_dir = Path(optimization_dir)
    base_result_directory = optimization_dir / "results"
    logger.debug(f"Loading results from {base_result_directory}")

    previous_paths, pending_paths = {}, {}
    for config_dir in base_result_directory.iterdir():
        if not config_dir.is_dir():
            continue
        config_id = config_dir.name[len("config_") :]
        config_file = config_dir / f"config{serializer.SUFFIX}"
        result_file = config_dir / f"result{serializer.SUFFIX}"

        if non_empty_file(result_file):
            previous_paths[config_id] = (config_dir, config_file, result_file)
        elif non_empty_file(config_file):
            pending_paths[config_id] = (config_dir, config_file)
        else:
            existing_config = find_files(
                config_dir, ["config"], any_suffix=True, check_nonempty=True
            )
            if existing_config:
                existing_format = existing_config[0].suffix
                logger.warning(
                    f"Found directory {config_dir} with file {existing_config[0].name}. "
                    f"But function was called with the serializer for "
                    f"'{serializer.SUFFIX}' files, not '{existing_format}'."
                )
            else:
                # Should probably warn the user somehow about this, although it is not
                # dangerous
                logger.warning(
                    f"Removing {config_dir} as worker died during config sampling."
                )
                try:
                    shutil.rmtree(str(config_dir))
                except Exception:  # The worker doesn't need to crash for this
                    logger.exception(f"Can't delete {config_dir}")
    return previous_paths, pending_paths


def _read_config_result(result_dir: Path | str, serializer: YamlSerializer):
    result_dir = Path(result_dir)
    config = serializer.load_config(result_dir / "config")
    result = serializer.load(result_dir / "result")
    metadata = serializer.load(result_dir / "metadata")
    return ConfigResult(config, result, metadata)


def read(optimization_dir: Path | str, serializer=None, logger=None, do_lock=True):
    optimization_dir = Path(optimization_dir)

    if logger is None:
        logger = logging.getLogger("metahyper")

    # if do_lock:
    #     decision_lock_file = optimization_dir / ".decision_lock"
    #     decision_lock_file.touch(exist_ok=True)
    #     decision_locker = Locker(decision_lock_file, logger.getChild("_locker"))
    #     while not decision_locker.acquire_lock():
    #         time.sleep(2)

    if serializer is None:
        serializer = YamlSerializer()

    previous_paths, pending_paths = _load_sampled_paths(
        optimization_dir, serializer, logger
    )
    previous_results, pending_configs, pending_configs_free = {}, {}, {}

    for config_id, (config_dir, _, _) in previous_paths.items():
        previous_results[config_id] = _read_config_result(config_dir, serializer)

    for config_id, (config_dir, config_file) in pending_paths.items():
        pending_configs[config_id] = serializer.load_config(config_file)

        # config_lock_file = config_dir / ".config_lock"
        # config_locker = Locker(config_lock_file, logger.getChild("_locker"))
        # if config_locker.acquire_lock():
        #     pending_configs_free[config_id] = pending_configs[config_id]

    logger.debug(
        f"Read in {len(previous_results)} previous results and "
        f"{len(pending_configs)} pending evaluations "
        f"({len(pending_configs_free)} without a worker)"
    )
    logger.debug(
        f"Read in previous_results={previous_results}, "
        f"pending_configs={pending_configs}, "
        f"and pending_configs_free={pending_configs_free}, "
    )

    # if do_lock:
    #     decision_locker.release_lock()
    return previous_results, pending_configs, pending_configs_free

def read_single(optimization_dir: Path | str, serializer=None, logger=None, do_lock=True):
    optimization_dir = Path(optimization_dir)

    if logger is None:
        logger = logging.getLogger("metahyper")

    if serializer is None:
        serializer = YamlSerializer()

    previous_paths, pending_paths = _load_sampled_paths(
        optimization_dir, serializer, logger
    )
    previous_results, pending_configs, pending_configs_free = {}, {}, {}

    for config_id, (config_dir, _, _) in previous_paths.items():
        previous_results[config_id] = _read_config_result(config_dir, serializer)

    for config_id, (config_dir, config_file) in pending_paths.items():
        pending_configs[config_id] = serializer.load_config(config_file)

    logger.debug(
        f"Read in {len(previous_results)} previous results and "
        f"{len(pending_configs)} pending evaluations "
        f"({len(pending_configs_free)} without a worker)"
    )
    logger.debug(
        f"Read in previous_results={previous_results}, "
        f"pending_configs={pending_configs}, "
        f"and pending_configs_free={pending_configs_free}, "
    )

    return previous_results, pending_configs, pending_configs_free, pending_paths


def _check_max_evaluations(
    optimization_dir,
    max_evaluations,
    serializer,
    logger,
    continue_until_max_evaluation_completed,
    previous_results,
    pending_configs,
):
    logger.debug("Checking if max evaluations is reached")
    # OH
    # previous_results, pending_configs, pending_configs_free = read(
    #     optimization_dir, serializer, logger
    # )
    evaluation_count = len(previous_results)

    # Taking into account pending evaluations
    if not continue_until_max_evaluation_completed:
        evaluation_count += len(pending_configs)

    return evaluation_count >= max_evaluations


def _sample_config(optimization_dir, sampler, serializer, logger, pre_load_hooks,
                   previous_results, pending_configs, pending_paths_free):

    base_result_directory = optimization_dir / "results"
    logger.debug(f"Previous results: {previous_results}")
    logger.debug(f"Pending configs: {pending_configs}")
    logger.debug(f"Pending paths: {pending_paths_free}")

    logger.debug("Sampling a new configuration")

    for hook in pre_load_hooks:
        # executes operations on the sampler before setting its state
        # can be used for setting custom constraints on the optimizer state
        # for example, can be used to input custom grid of configs, meta learning
        # information for surrogate building, any non-stationary auxiliary information
        sampler = hook(sampler)

    if not pending_paths_free:
        sampler.load_results(previous_results, pending_configs)
        config, config_id, previous_config_id = sampler.get_config_and_ids()
    else:
        # Handle Pending configuration before moving on to sampling new
        # This will be the case when a run is terminated prematurely and
        # Still had a sampled but not yet evaluated config from the previous run
        config_id = list(pending_paths_free.keys())[0]
        # Read into dictionary form since the eval function expects only a dictionary
        config = pending_configs.pop(config_id).hp_values()
        pipeline_directory = base_result_directory / f"config_{config_id}"
        prev_config_directory = pipeline_directory / "previous_config.id"
        if prev_config_directory.exists():
            previous_config_id = prev_config_directory.read_text()
            previous_pipeline_directory = base_result_directory / f"config_{previous_config_id}"
        else:
            previous_pipeline_directory = None
        (config_dir, _) = pending_paths_free.pop(config_id)

        logger.warning(f"Found a not yet evaluated config in {config_dir}\n"
                       f"Evaluating this config before starting the optimizer")
        return (config_id,
                config,
                pipeline_directory,
                previous_pipeline_directory)

    pipeline_directory = base_result_directory / f"config_{config_id}"
    pipeline_directory.mkdir(exist_ok=True)

    # write some extra data per configuration if the optimizer has any
    if hasattr(sampler, "evaluation_data"):
        sampler.evaluation_data.write_all(pipeline_directory)

    if previous_config_id is not None:
        previous_config_id_file = pipeline_directory / "previous_config.id"
        previous_config_id_file.write_text(previous_config_id)  # TODO: Get rid of this
        serializer.dump(
            {"time_sampled": time.time(), "previous_config_id": previous_config_id},
            pipeline_directory / "metadata",
        )
        previous_pipeline_directory = Path(
            base_result_directory, f"config_{previous_config_id}"
        )
    else:
        serializer.dump({"time_sampled": time.time()}, pipeline_directory / "metadata")
        previous_pipeline_directory = None

    # We want this to be the last action in sampling to catch potential crashes
    serializer.dump(config, pipeline_directory / "config")

    logger.debug(f"Sampled config {config_id}")
    return (
        config_id,
        config,
        pipeline_directory,
        previous_pipeline_directory,
    )


def _evaluate_config(
    config_id,
    config,
    pipeline_directory,
    evaluation_fn,
    previous_pipeline_directory,
    logger,
):
    if isinstance(config, Configuration):
        config = config.hp_values()
    config = deepcopy(config)
    logger.info(f"Start evaluating config {config_id}")
    try:
        # If pipeline_directory and previous_pipeline_directory are included in the
        # signature we supply their values, otherwise we simply do nothing.
        evaluation_fn_params = inspect.signature(evaluation_fn).parameters
        directory_params = []
        if "pipeline_directory" in evaluation_fn_params:
            directory_params.append(pipeline_directory)
        elif "working_directory" in evaluation_fn_params:
            warnings.warn(
                "the argument: 'working_directory' is deprecated. "
                "In the function: '{}', please, "
                "use 'pipeline_directory' instead. "
                "version==0.5.5".format(evaluation_fn.__name__),
                DeprecationWarning,
                stacklevel=2,
            )
            directory_params.append(pipeline_directory)

        if "previous_pipeline_directory" in evaluation_fn_params:
            directory_params.append(previous_pipeline_directory)
        elif "previous_working_directory" in evaluation_fn_params:
            warnings.warn(
                "the argument: 'previous_working_directory' is deprecated. "
                "In the function: '{}', please,  "
                "use 'previous_pipeline_directory' instead. "
                "version==0.5.5".format(evaluation_fn.__name__),
                DeprecationWarning,
                stacklevel=2,
            )
            directory_params.append(previous_pipeline_directory)

        result = evaluation_fn(
            *directory_params,
            **config,
        )

        # Ensuring the result have the correct format that can be exploited by other functions
        if isinstance(result, dict):
            try:
                result["loss"] = float(result["loss"])
            except KeyError as e:
                raise ValueError("The loss should value should be provided") from e
            except (TypeError, ValueError) as e:
                raise ValueError("The loss should be a float") from e
        else:
            try:
                result = float(result)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "The evaluation result should be a dictionnary or a float"
                ) from e
            result = {"loss": result}
    except Exception:
        logger.exception(
            f"An error occured during evaluation of config {config_id}: " f"{config}."
        )
        result = "error"

    return result, {"time_end": time.time()}


def run(
    evaluation_fn,
    sampler: Sampler,
    sampler_info: dict,
    optimization_dir,
    max_evaluations_total=None,
    max_evaluations_per_run=None,
    continue_until_max_evaluation_completed=False,
    development_stage_id=None,
    task_id=None,
    logger=None,
    post_evaluation_hook=None,
    overwrite_optimization_dir=False,
    pre_load_hooks: List=[],
):
    serializer = YamlSerializer(sampler.load_config)
    if logger is None:
        logger = logging.getLogger("metahyper")

    if task_id is not None:
        optimization_dir = Path(optimization_dir) / f"task_{task_id}"
    if development_stage_id is not None:
        optimization_dir = Path(optimization_dir) / f"dev_{development_stage_id}"

    optimization_dir = Path(optimization_dir)
    if overwrite_optimization_dir and optimization_dir.exists():
        logger.warning("Overwriting working_directory")
        shutil.rmtree(optimization_dir)

    sampler_state_file = optimization_dir / ".optimizer_state.yaml"
    sampler_info_file = optimization_dir / ".optimizer_info.yaml"
    base_result_directory = optimization_dir / "results"
    base_result_directory.mkdir(parents=True, exist_ok=True)

    zip_filename = Path(optimization_dir / "results.zip")
    # Extract previous results to load if it exists
    if zip_filename.exists():
        #  and not any(Path(base_result_directory).iterdir()):
        shutil.unpack_archive(zip_filename, base_result_directory, "zip")
        zip_filename.unlink()

    # decision_lock_file = optimization_dir / ".decision_lock"
    # decision_lock_file.touch(exist_ok=True)
    # decision_locker = Locker(decision_lock_file, logger.getChild("_locker"))

    _process_sampler_info(
        serializer, sampler_info, sampler_info_file, None, logger
    )

    previous_results, pending_configs, _, pending_paths_free = read_single(
        optimization_dir, serializer, logger, do_lock=False
    )

    evaluations_in_this_run = 0

    # Read the state file from the previous run
    # with sampler.using_state(sampler_state_file, serializer):
    if sampler_state_file.exists():
        sampler.load_state(serializer.load(sampler_state_file))
    while True:
        if max_evaluations_total is not None and _check_max_evaluations(
            optimization_dir,
            max_evaluations_total,
            serializer,
            logger,
            continue_until_max_evaluation_completed,
            previous_results,
            pending_configs
        ):
            logger.info("Maximum total evaluations is reached, shutting down")
            break

        if (
            max_evaluations_per_run is not None
            and evaluations_in_this_run >= max_evaluations_per_run
        ):
            logger.info("Maximum evaluations per run is reached, shutting down")
            break

        # if decision_locker.acquire_lock():
        # try:

        if sampler.budget is not None:
            if sampler.used_budget >= sampler.budget:
                logger.info("Maximum budget reached, shutting down")
                break
        (
            config_id,
            config,
            pipeline_directory,
            previous_pipeline_directory,
        ) = _sample_config(
            optimization_dir, sampler, serializer, logger, pre_load_hooks,
            previous_results, pending_configs, pending_paths_free
        )
        # Storing the config details in ConfigInRun
        ConfigInRun.store_in_run_data(
            config,
            config_id,
            pipeline_directory,
            previous_pipeline_directory,
            optimization_dir,
        )

        #     config_lock_file = pipeline_directory / ".config_lock"
        #     config_lock_file.touch(exist_ok=True)
        #     config_locker = Locker(config_lock_file, logger.getChild("_locker"))
        #     config_lock_acquired = config_locker.acquire_lock()
        # finally:
        #     decision_locker.release_lock()

        # if config_lock_acquired:
    #     try:
        # 1. First, we evaluate the config
        result, metadata = _evaluate_config(
            config_id,
            config,
            pipeline_directory,
            evaluation_fn,
            previous_pipeline_directory,
            logger,
        )

        # 2. Then, we now dump all information to disk:
        serializer.dump(result, pipeline_directory / "result")

        if result != "error":
            # Updating the global budget
            if "cost" in result:
                eval_cost = float(result["cost"])
                account_for_cost = result.get("account_for_cost", True)
                if account_for_cost:
                    # with decision_locker.acquire_force(time_step=1):
                        # with sampler.using_state(
                        #     sampler_state_file, serializer
                        # ):
                    sampler.used_budget += eval_cost

                metadata["budget"] = {
                    "max": sampler.budget,
                    "used": sampler.used_budget,
                    "eval_cost": eval_cost,
                    "account_for_cost": account_for_cost,
                }
            elif sampler.budget is not None:
                raise ValueError(
                    "The evaluation function result should contain "
                    "a 'cost' field when used with a budget"
                )

        config_metadata = serializer.load(pipeline_directory / "metadata")
        config_metadata.update(metadata)
        serializer.dump(config_metadata, pipeline_directory / "metadata")

        # Update previous_results manually
        config_ = serializer.load_config(pipeline_directory / "config")
        config_res = ConfigResult(config_, result, config_metadata)
        previous_results[config_id] = config_res

        # 3. Anything the user might want to do after the evaluation
        if post_evaluation_hook is not None:
            post_evaluation_hook(
                config, config_id, pipeline_directory, result, logger
            )
        else:
            logger.info(f"Finished evaluating config {config_id}")

        # Write Optimizer state file
        serializer.dump(sampler.get_state(), sampler_state_file)
        # Zip all the results
        # zip_filename = Path(str(base_result_directory) + ".zip")
        with zipfile.ZipFile(zip_filename, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for entry in base_result_directory.rglob("*"):
                zip_file.write(entry, entry.relative_to(base_result_directory))
        # remove results
        shutil.rmtree(base_result_directory)
        # create empty results directory
        base_result_directory.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_filename, "a", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in base_result_directory.rglob("*"):
            zip_file.write(entry, entry.relative_to(base_result_directory))
    # remove results
    shutil.rmtree(base_result_directory)
        # finally:
        #     config_locker.release_lock()