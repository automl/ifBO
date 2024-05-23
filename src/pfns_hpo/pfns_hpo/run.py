import contextlib
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, List
import ConfigSpace as CS
import hydra
import numpy as np
import pandas as pd
import yaml
from gitinfo import gitinfo
from omegaconf import OmegaConf

from mfpbench import Benchmark
from mfpbench.result import Result
from mfpbench.taskset_tabular.benchmark import TaskSetTabularBenchmark

import neps
from neps.search_spaces.search_space import SearchSpace, pipeline_space_from_configspace

from .plot3D import Plotter3D

logger = logging.getLogger("pfns_hpo.run")

# NOTE: If editing this, please look for MIN_SLEEP_TIME
# in `read_results.py` and change it there too
MIN_SLEEP_TIME = 10  # 10s hopefully is enough to simulate wait times for metahyper

# Use this environment variable to force overwrite when running
OVERWRITE = False  # bool(os.environ.get("MF_EXP_OVERWRITE", False))

print(f"{'='*50}\noverwrite={OVERWRITE}\n{'='*50}")

BOHB_MAX_EVALS = 10  # number of HB brackets
NEPS_MF_EI_MAX_EVALS = 1000  # number of total function evaluations for multi-fidelity
NEPS_MF_MAX_EVALS = 400  # number of total function evaluations for multi-fidelity
NEPS_SF_MAX_EVALS = 200 # number of total function evaluations for single-fidelity

SET_BOUNDS_FROM_TABLE_FLAG = False  # if True, sets search space bounds from table values

def _set_seeds(seed):
    random.seed(seed)  # important for NePS optimizers
    np.random.seed(seed)  # important for NePS optimizers
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(seed)
    # tf.random.set_seed(seed)

def process_taskset_mfpbench_with_step_0_prior(
    benchmark: TaskSetTabularBenchmark,
    loss_name_at_step_0_to_normalize_with: str = "valid1_loss",
    metrics_to_normalize_and_clamp: tuple[str, ...] = (
        "train_loss",
        "valid1_loss",
        "valid2_loss",
        "test_loss",
    ),
    drop_step_0: bool = True,
) -> TaskSetTabularBenchmark:
    """Normalize metrics of benchmark with the step 0 median huerisitc.

    To handle the case of unknown upper bounds, we need to normalize the results
    returned to algorithms in a (0, 1) range. We do this through clipping to
    some hueristic upper bound and then normalizing w.r.t. to this upper bound.

    In practice, a practioner could provide this upper bound value from the prior,
    either from rapid experimentation or previous experimental trials. They would
    then normalize the results before feeding it the hpo algorithm. For convenience
    sake, we apply this hueristic across the table using this prior knowledge.

    To generalize across tables, where peeking at the maximum loss value could be
    considered leaking information, we apply a heuristic that we believe matches
    the above intuitions, namely we use the median loss at step 0 as the upper
    bound. This corresponds to the median loss of the initial random configuration
    and is a reasonable heuristic for the upper bound. Notably, this does not look
    deep into the table to perform normalizations.

    Args:
        benchmark: The benchmark to normalize
        loss_name_at_step_0_to_normalize_with: The loss column from which
            to apply the hueristic and get the upper bound. By default,
            this is the validation loss used in the benchmark.
        metrics_to_normalize_and_clamp: The metric columns to normalize and clamp
            according to the upper bound. By default, this applies to all
            loss columns.
        drop_step_0: Whether to drop the step 0 values after normalization. By default,
            this will drop the step 0 values. These are random initializations and would
            generally not be reported to a HPO algorithm.
    """
    table = benchmark.table

    # Make sure that the table has a step column first
    if "step" not in table.columns:
        raise ValueError("Benchmark does not have 'step' in columns")

    # Make sure benchmarks contain the random initializations we use for the heuristic
    values = table[table["step"] == 0]
    assert values.index.is_unique, f"Step 0 not unique across configs?\n{values.index}"

    # Make sure that the loss column we are using to normalize with is non-negative
    # and corresponds to a regular log-loss, where the known theoretical bound is 0.
    # This is so the min-max normalization is well-defined and we can rely on the
    # theoretical bound to be 0 for normalization.
    if table[loss_name_at_step_0_to_normalize_with].min() < 0:
        raise ValueError(
            f"Benchmark has negative loss in '{loss_name_at_step_0_to_normalize_with}',"
            " normalization can not be applied naively",
        )

    # Get the median loss at step 0 as the hueuristic upper bound
    median_loss_to_use_as_prior = values[loss_name_at_step_0_to_normalize_with].median()

    # Include the normalizing bound in the table
    # NOTE: This causes a crash with neps and isn't essneital
    #table[
        #f"normalizing_bound_from_{loss_name_at_step_0_to_normalize_with}"
    #] = median_loss_to_use_as_prior

    for metric in metrics_to_normalize_and_clamp:
        # Make sure to store the corresponding non-normalized metric
        # NOTE: This causes a crash with neps and isn't essneital
        # table[f"{metric}_unnormalized"] = table[metric].copy()

        # Clip according to the median loss at step 0
        table[metric] = table[metric].clip(lower=0, upper=median_loss_to_use_as_prior)

        # And then normalize w.r.t. to this upper bound
        table[metric] = table[metric] / median_loss_to_use_as_prior

    if drop_step_0:
        # Select all rows that are not step 0
        table = table[table["step"] != 0]

        # Decrease the epoch number by one for uniformity
        table = table.reset_index()
        table["epoch"] = table["epoch"] - 1
        table = table.set_index(["id", "epoch"]).sort_index()

        # Decrease the fidelity range by 1
        lower, upper, step = benchmark.fidelity_range
        benchmark.fidelity_range = (lower, upper - 1, step)

    benchmark.table = table

    if benchmark.table.isna().any().any():
        print(benchmark.table.isna().any())
        raise ValueError("There should not be an na's left")

    return benchmark

def process_mfpbench_trajectories(res: List[Result]) -> dict:
    valid_errors = []
    test_errors = []
    fidelity = []

    for r in res:
        valid_errors.append(r.error)
        test_errors.append(r.test_error)
        fidelity.append(r.fidelity)

    result = dict(
        valid=valid_errors,
        test=test_errors,
        fidelity=fidelity,
    )
    min_valid_seen = min(valid_errors)
    min_test_seen = min(test_errors)

    return result, min_valid_seen, min_test_seen

def run_dpl(args):
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.absolute() / "DPL"))
    print(sys.path)

    from custom_framework import Framework

    delattr(args.benchmark.api, "step_size")
    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)

    # Maybe apply step 0 median normalization, read docstring of func for more
    if (
        isinstance(benchmark, TaskSetTabularBenchmark)
        and args.benchmark.get("apply_user_prior_step_0_median_normalized", False) is True
    ):
        benchmark = process_taskset_mfpbench_with_step_0_prior(
            benchmark=benchmark,
            drop_step_0=args.benchmark.get("drop_epoch_0", True)
        )

    min_value, max_value = 0.0, 1.0 # minimization
    budget_limit = NEPS_MF_EI_MAX_EVALS
    fantasize_step = 1

    hp_names = []
    log_indicator = []
    categorical_indicator = []
    for hp in benchmark.space.get_hyperparameters():
        hp_names.append(hp.name)
        if hasattr(hp, "log"):
            log_indicator.append(hp.log)
        else:
            log_indicator.append(False)
        categorical_indicator.append(isinstance(hp, CS.CategoricalHyperparameter))

    configs = benchmark.configs

    config_keys = [int(_) for _ in configs.keys()]
    configs_values = [[configs[str(id)][cfg] for cfg in hp_names] for id in config_keys]
    configs_values = np.array(configs_values)
    
    minimization_metric = True
    
    framework = Framework(
        dataset_name=benchmark.name,
        budget_limit=budget_limit,
        fantasize_step=fantasize_step,
        output_dir="",
        ensemble_size=5,
        nr_epochs=250,
        seed=args.seed,
        benchmark_max_value=max_value,
        benchmark_min_value=min_value,
        benchmark_categorical_indicator=categorical_indicator,
        benchmark_log_indicator=log_indicator,
        benchmark_hp_names=hp_names,
        benchmark_minimization_metric=minimization_metric,
        benchmark_max_budget=benchmark.fidelity_range[1],
        benchmark_configs_values=configs_values,
    )

    evaluated_configs = dict()
    surrogate_budget = 0

    best_value = np.inf if minimization_metric else 0 # 0 is not a good default

    run_result = []
    while surrogate_budget < framework.total_budget:
        start_time = time.time()

        hp_index, budget = framework.surrogate.suggest()
        surrogate_budget += 1

        config = config_keys[hp_index]

        full_trajectory = benchmark.trajectory(config)
        trajectory_to_query = [r for r in full_trajectory if r.fidelity <= budget]

        result = trajectory_to_query[-1]
        max_fidelity_result = full_trajectory[-1]

        # best seen till the fidelity specified
        _result, min_valid_seen, min_test_seen = process_mfpbench_trajectories(trajectory_to_query)
        # best seen ever for the config till max fidelity
        _, min_valid_ever, min_test_ever = process_mfpbench_trajectories(full_trajectory)

        hp_curve = _result["valid"]

        framework.surrogate.observe(hp_index, budget, hp_curve)
        time_duration = time.time() - start_time

        if hp_index in evaluated_configs:
            previous_budget = evaluated_configs[hp_index]
        else:
            previous_budget = 0

        end = time.time()

        run_result.append({
            "loss": result.error,
            "cost": result.cost,
            "info_dict": {
                "cost": result.cost,
                "val_score": result.val_score,
                "test_score": result.test_score,
                "fidelity": result.fidelity,
                "continuation_fidelity": None,
                "start_time": start_time,
                "end_time": end,  # + fidelity,
                "max_fidelity_loss": float(max_fidelity_result.error),
                "max_fidelity_cost": float(max_fidelity_result.cost),
                "process_id": os.getpid(),
                "min_valid_seen": min_valid_seen,
                "min_test_seen": min_test_seen,
                "min_valid_ever": min_valid_ever,
                "min_test_ever": min_test_ever,
                "learning_curve": _result["valid"],
                "learning_curves": _result,
                "config_id": f"{hp_index}_{result.fidelity - 1}"   # -1 to equate to neps budgets
            },
        })
    # end of while

    from pfns_hpo.utils.plotting_utils import flatten_dict

    output_path = Path().cwd() / "arlind_root_directory"
    output_path.mkdir(parents=True, exist_ok=True)
    flattened_data = pd.DataFrame.from_dict(
        [flatten_dict(item, 'result') for item in run_result]
    )
    flattened_data.to_csv(output_path / "run_data.csv")

    plotter = Plotter3D()
    plotter.plot(flattened_data, Path().cwd())

    return


def run_dyhpo(args):
    import sys
    import numpy as np
    sys.path.append(str(Path(__file__).parent.parent.parent.absolute() / "DyHPO"))
    # print(f"\nPATH: {sys.path}\n")
    from hpo_method import DyHPOAlgorithm

    # Added the type here just for editors to be able to get a quick view
    delattr(args.benchmark.api, "step_size")
    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)

    # Maybe apply step 0 median normalization, read docstring of func for more
    if (
        isinstance(benchmark, TaskSetTabularBenchmark)
        and args.benchmark.get("apply_user_prior_step_0_median_normalized", False) is True
    ):
        benchmark = process_taskset_mfpbench_with_step_0_prior(
            benchmark=benchmark,
            drop_step_0=args.benchmark.get("drop_epoch_0", True)
        )

    bench_is_tabular = (
        True if hasattr(args.benchmark, "tabular") and args.benchmark.tabular else False
    )
    if not bench_is_tabular:
        raise RuntimeError("This algorithm can only handle tabular benchmarks!")

    configs = benchmark.configs

    hp_names = []
    log_indicator = []
    categorical_indicator = []
    for hp in benchmark.space.get_hyperparameters():
        hp_names.append(hp.name)
        log_indicator.append(hp.log)
        assert not isinstance(hp, CS.OrdinalHyperparameter), "Ordinal HPs not supported by the original implementation!"

    config_keys = [int(_) for _ in configs.keys()]
    configs_values = [[configs[str(id)][cfg] for cfg in hp_names] for id in config_keys]
    configs_values = np.array(configs_values)

    # TODO: align with neps
    budget_limit = NEPS_MF_EI_MAX_EVALS
    fantasize_step = 1
    evaluated_configs = dict()

    method_budget = 0

    dyhpo_surrogate = DyHPOAlgorithm(
        hp_candidates=configs_values,
        log_indicator=log_indicator,
        seed=args.seed,
        max_benchmark_epochs=benchmark.fidelity_range[1],
        fantasize_step=1,
        minimization=True,  # TODO: correctly
        total_budget=budget_limit,
        dataset_name=benchmark.name,
        output_path="",
        surrogate_config=None, # default config
        verbose=True,
    )

    run_result = []
    while method_budget < budget_limit:
        start = time.time()

        hp_index, budget = dyhpo_surrogate.suggest()

        config = config_keys[hp_index]

        full_trajectory = benchmark.trajectory(config)
        trajectory_to_query = [r for r in full_trajectory if r.fidelity <= budget]

        result = trajectory_to_query[-1]
        max_fidelity_result = full_trajectory[-1]

        # best seen till the fidelity specified
        _result, min_valid_seen, min_test_seen = process_mfpbench_trajectories(trajectory_to_query)
        # best seen ever for the config till max fidelity
        _, min_valid_ever, min_test_ever = process_mfpbench_trajectories(full_trajectory)

        dyhpo_surrogate.observe(hp_index, budget, _result["valid"])

        budget_cost = 0
        if hp_index in evaluated_configs:
            previous_budget = evaluated_configs[hp_index]
            budget_cost = budget - previous_budget
            evaluated_configs[hp_index] = budget
        else:
            budget_cost = fantasize_step

        method_budget += budget_cost
        logger.info(f"Remaining budget: {budget_limit - method_budget}")

        end = time.time()

        run_result.append({
            "loss": result.error,
            "cost": result.cost,
            "info_dict": {
                "cost": result.cost,
                "val_score": result.val_score,
                "test_score": result.test_score,
                "fidelity": result.fidelity,
                "continuation_fidelity": None,
                "start_time": start,
                "end_time": end,  # + fidelity,
                "max_fidelity_loss": float(max_fidelity_result.error),
                "max_fidelity_cost": float(max_fidelity_result.cost),
                "process_id": os.getpid(),
                "min_valid_seen": min_valid_seen,
                "min_test_seen": min_test_seen,
                "min_valid_ever": min_valid_ever,
                "min_test_ever": min_test_ever,
                "learning_curve": _result["valid"],
                "learning_curves": _result,
                "config_id": f"{hp_index}_{result.fidelity - 1}"   # -1 to equate to neps budgets
            },
        })
    # end of while

    from pfns_hpo.utils.plotting_utils import flatten_dict

    flattened_data = pd.DataFrame.from_dict(
        [flatten_dict(item, 'result') for item in run_result]
    )
    output_path = Path().cwd() / "arlind_root_directory"
    output_path.mkdir(parents=True, exist_ok=True)
    flattened_data.to_csv(output_path / "run_data.csv")

    plotter = Plotter3D()
    plotter.plot(flattened_data, Path().cwd())

    return


def run_hpbandster(args):
    import uuid

    import ConfigSpace
    import hpbandster.core.nameserver as hpns
    import hpbandster.core.result as hpres
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers.bohb import BOHB
    from hpbandster.optimizers.hyperband import HyperBand
    from mfpbench import Benchmark

    # HpBandster based algos take no step_size as input
    delattr(args.benchmark.api, "step_size")

    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)  # type: ignore

    # Maybe apply step 0 median normalization, read docstring of func for more
    if (
        isinstance(benchmark, TaskSetTabularBenchmark)
        and args.benchmark.get("apply_user_prior_step_0_median_normalized", False) is True
    ):
        benchmark = process_taskset_mfpbench_with_step_0_prior(
            benchmark=benchmark,
            drop_step_0=args.benchmark.get("drop_epoch_0", True)
        )

    # CRUCIAL check to determine if the benchmark is tabular (list of configs)
    bench_is_tabular = (
        True if hasattr(args.benchmark, "tabular") and args.benchmark.tabular else False
    )

    # # Added the type here just for editors to be able to get a quick view
    # benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)

    def compute(**config: Any) -> dict:
        fidelity = config["budget"]
        config = config["config"]

        if bench_is_tabular:  # declared in parent scope
            # IMPORTANT to handle tabular benchmarks to query using only IDs
            if "lcbench" in args.benchmark.name:
                config = int(config["id"])
            # TODO: handle other tabular benchmarks

        # transform to Ordinal HPs back
        for hp_name, hp in benchmark.space.items():
            if isinstance(hp, ConfigSpace.OrdinalHyperparameter):
                config[hp_name] = hp.sequence[config[hp_name] - 1]

        # The design below of multiple calls to the benchmark only makes sense
        # #in the context of surrogate/tabular benchmarks, where we do not actually
        # need to run the model being queried.
        full_trajectory = benchmark.trajectory(config)
        trajectory_to_query = [r for r in full_trajectory if r.fidelity <= fidelity]

        result = trajectory_to_query[-1]
        max_fidelity_result = full_trajectory[-1]

        # best seen till the fidelity specified
        _result, min_valid_seen, min_test_seen = process_mfpbench_trajectories(trajectory_to_query)
        # best seen ever for the config till max fidelity
        _, min_valid_ever, min_test_ever = process_mfpbench_trajectories(full_trajectory)

        # we need to cast to float here as serpent will break on np.floating that might
        # come from a benchmark (LCBench)
        return {
            "loss": float(result.error),
            "cost": float(result.cost),
            "info": {
                "cost": float(result.cost),
                "val_score": float(result.val_score),
                "test_score": float(result.test_score),
                "fidelity": float(result.fidelity),
                "max_fidelity_loss": float(max_fidelity_result.error),
                "max_fidelity_cost": float(max_fidelity_result.cost),
                "process_id": os.getpid(),
                "min_valid_seen": min_valid_seen,
                "min_test_seen": min_test_seen,
                "min_valid_ever": min_valid_ever,
                "min_test_ever": min_test_ever,
                "learning_curve": _result["valid"],
                "learning_curve_train": _result["train"],
                "learning_curve_test": _result["test"],
                # "learning_curves": _result,  # TODO: fix -> cannot parse dict here
            }
        }

    lower, upper, _ = benchmark.fidelity_range
    fidelity_name = benchmark.fidelity_name
    benchmark_configspace = benchmark.space

    # BOHB does not accept Ordinal HPs
    bohb_configspace = ConfigSpace.ConfigurationSpace(
        name=benchmark_configspace.name, seed=args.seed
    )

    for hp_name, hp in benchmark_configspace.items():
        if isinstance(hp, ConfigSpace.OrdinalHyperparameter):
            int_hp = ConfigSpace.UniformIntegerHyperparameter(
                hp_name, lower=1, upper=len(hp.sequence)
            )
            bohb_configspace.add_hyperparameters([int_hp])
        else:
            bohb_configspace.add_hyperparameters([hp])

    logger.info(f"Using configspace: \n {benchmark_configspace}")
    logger.info(f"Using fidelity: \n {fidelity_name} in {lower}-{upper}")

    max_evaluations_total = BOHB_MAX_EVALS

    run_id = str(uuid.uuid4())
    NS = hpns.NameServer(
        run_id=run_id, port=0, working_directory="hpbandster_root_directory"
    )
    ns_host, ns_port = NS.start()

    hpbandster_worker = Worker(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id)
    hpbandster_worker.compute = compute
    hpbandster_worker.run(background=True)

    result_logger = hpres.json_result_logger(
        directory="hpbandster_root_directory", overwrite=True
    )
    hpbandster_config = {
        "eta": 3,
        "min_budget": lower,
        "max_budget": upper,
        "run_id": run_id,
    }

    if "model" in args.algorithm and args.algorithm.model:
        hpbandster_cls = BOHB
    else:
        hpbandster_cls = HyperBand

    hpbandster_optimizer = hpbandster_cls(
        configspace=bohb_configspace,
        nameserver=ns_host,
        nameserver_port=ns_port,
        result_logger=result_logger,
        **hpbandster_config,
    )

    logger.info("Starting run...")
    res = hpbandster_optimizer.run(n_iterations=max_evaluations_total)

    hpbandster_optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    logger.info(f"A total of {len(id2config.keys())} queries.")
# end of run_hpbandster


def run_neps(args):
    import shutil

    # unpack the results.zip if it exists, then delete it
    # moved to neps
    # if os.path.exists("neps_root_directory") and not os.path.exists("neps_root_directory/results") and os.path.exists("neps_root_directory/results.zip"):
    #     shutil.unpack_archive("neps_root_directory/results.zip", "neps_root_directory/", "zip")
    #     os.remove("neps_root_directory/results.zip")

    # If algorithm has a step size
    # if hasattr(args.algorithm.searcher_kwargs, "step_size"):
    #     # Then get the step_size from the benchmark, if defined
    #     if hasattr(args.benchmark.api, "step_size"):
    #         args.algorithm.searcher_kwargs.update(dict(step_size=args.benchmark.api.step_size))
    # # Then delete the step_size attribute to avoid unintended behaviours

    # TODO: for fair experimentation, make this a benchmark property overriding algo
    delattr(args.benchmark.api, "step_size")

    # # If algorithm has a step size
    # if hasattr(args.algorithm.searcher, "step_size"):
    #     # Then get the step_size from the benchmark, if defined
    #     if hasattr(args.benchmark.api, "step_size"):
    #         args.algorithm.searcher.update(dict(step_size=args.benchmark.api.step_size))
    # # Then delete the step_size attribute to avoid unintended behaviours
    # delattr(args.benchmark.api, "step_size")

    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)  # type: ignore

    # Maybe apply step 0 median normalization, read docstring of func for more
    if (
        isinstance(benchmark, TaskSetTabularBenchmark)
        and args.benchmark.get("apply_user_prior_step_0_median_normalized", False) is True
    ):
        print(f"PREPROCESSING {benchmark.name} with 'apply_user_prior_step_0_median_normalized'")
        drop_0_epoch = args.benchmark.get("drop_epoch_0", True)
        print(f"PREPROCESSING {benchmark.name} with '{drop_0_epoch=}'")
        benchmark = process_taskset_mfpbench_with_step_0_prior(
            benchmark=benchmark,
            drop_step_0=drop_0_epoch,
        )

    # CRUCIAL check to determine if the benchmark is tabular (list of configs)
    bench_is_tabular = (
        True if hasattr(args.benchmark, "tabular") and args.benchmark.tabular else False
    )
    def run_pipeline(previous_pipeline_directory: Path, **config: Any) -> dict:
        start = time.time()
        if benchmark.fidelity_name in config:
            fidelity = config.pop(benchmark.fidelity_name)
        else:
            fidelity = benchmark.fidelity_range[1]

        if bench_is_tabular:  # declared in parent scope
            # IMPORTANT to handle tabular benchmarks to query using only IDs
            # if "lcbench" in args.benchmark.name:
            if "tabular" in args.benchmark.name:
                config = int(config["id"])
            # TODO: handle other tabular benchmarks

        full_trajectory = benchmark.trajectory(config)
        trajectory_to_query = [r for r in full_trajectory if r.fidelity <= fidelity]

        result = trajectory_to_query[-1]
        max_fidelity_result = full_trajectory[-1]

        # best seen till the fidelity specified
        _result, min_valid_seen, min_test_seen = process_mfpbench_trajectories(trajectory_to_query)
        # best seen ever for the config till max fidelity
        _, min_valid_ever, min_test_ever = process_mfpbench_trajectories(full_trajectory)

        end = time.time()

        return {
            "loss": result.error,
            "cost": result.cost,
            "info_dict": {
                "cost": result.cost,
                "val_score": result.val_score,
                "test_score": result.test_score,
                "fidelity": result.fidelity,
                "continuation_fidelity": None,
                "start_time": start,
                "end_time": end,  # + fidelity,
                "max_fidelity_loss": float(max_fidelity_result.error),
                "max_fidelity_cost": float(max_fidelity_result.cost),
                "process_id": os.getpid(),
                "min_valid_seen": min_valid_seen,
                "min_test_seen": min_test_seen,
                "min_valid_ever": min_valid_ever,
                "min_test_ever": min_test_ever,
                "learning_curve": _result["valid"],
                "learning_curves": _result,
            },
        }

    pipeline_space = {
        "search_space": benchmark.space
    }
    lower, upper, _ = benchmark.fidelity_range
    fidelity_name = benchmark.fidelity_name
    if "mf" in args.algorithm and args.algorithm.mf:
        if isinstance(lower, float):
            fidelity_param = neps.FloatParameter(
                lower=lower, upper=upper, is_fidelity=True
            )
        else:
            fidelity_param = neps.IntegerParameter(
                lower=lower, upper=upper, is_fidelity=True
            )
        pipeline_space = {**pipeline_space, **{fidelity_name: fidelity_param}}
        logger.info(f"Using fidelity space: \n {fidelity_param}")
    logger.info(f"Using search space: \n {pipeline_space}")

    if "mf" in args.algorithm and args.algorithm.mf:
        max_evaluations_total = NEPS_MF_MAX_EVALS if args.algorithm.sh_based else NEPS_MF_EI_MAX_EVALS
    else:
        max_evaluations_total = NEPS_SF_MAX_EVALS

    # placeholder pre_load hook
    def set_grid_table_space(
        obj,
        **kwargs
    ) -> Any:
        return obj

    # snippet to handle tabular benchmarks
    if bench_is_tabular:
        # extracting and processing the tabular data and raw space
        _table = preprocess_tabular(args.benchmark.name, benchmark.table)
        # updates the pipeline_space to be only config IDs mapping to tabular data
        pipeline_space = {
            "id": neps.IntegerParameter(
                lower=_table.index.min(), upper=_table.index.max()
            )
        }
        # include the fidelity in the spaces
        if "mf" in args.algorithm and args.algorithm.mf:
            pipeline_space.update({fidelity_name: fidelity_param})
            # include fidelity in the raw benchmark space
            if isinstance(benchmark.space, CS.ConfigurationSpace):
                _space = pipeline_space_from_configspace(benchmark.space)
                _space.update({fidelity_name: fidelity_param})
                benchmark.space = SearchSpace(**_space)
            elif isinstance(benchmark.space, dict):
                benchmark.space.update({fidelity_name: fidelity_param})
            elif isinstance(benchmark.space, SearchSpace):
                benchmark.space.add_hyperparameter(name=fidelity_name, hp=fidelity_param)
            else:
                raise ValueError("Unknown benchmark space type!")
        else:
            benchmark.space = SearchSpace(**pipeline_space_from_configspace(benchmark.space))

        # (re-)defines a pre_load_hook to handle tabular data explicitly
        # CRUCIAL for tabular benchmarks with a fixed list of configs
        def set_grid_table_space(
            # overwrites the placeholder in the parent scope
            obj,
            table: pd.DataFrame | pd.Series = _table,
            space: CS.ConfigurationSpace = benchmark.space,
        ) -> Any:
            # both table and space are required to handle tabular spaces
            obj.pipeline_space.set_custom_grid_space(table, space)
            if SET_BOUNDS_FROM_TABLE_FLAG:
                obj = set_bounds_from_table(obj, table, space)
            return obj
    # end of tabular check block

    print("MAX EVALUATIONS:", max_evaluations_total)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="neps_root_directory",
        # TODO: figure out how to pass runtime budget and if metahyper internally
        #  calculates continuation costs to subtract from optimization budget
        # **budget_args,
        max_evaluations_total=max_evaluations_total,
        searcher=args.algorithm.name,  # hydra.utils.instantiate(args.algorithm.searcher, _partial_=True),
        searcher_path=Path(__file__).parent / "configs" / "algorithm",
        overwrite_working_directory=OVERWRITE,
        pre_load_hooks=[set_grid_table_space],  # crucial in allowing tabular grid access
        post_run_summary=True,  # important for efficient plotting
    )
    # end of run_neps

    # zip results folder
    # moved to neps
    # shutil.make_archive("neps_root_directory/results", "zip", "neps_root_directory/", "results")
    # shutil.rmtree("neps_root_directory/results")

    if "mf" in args.algorithm and args.algorithm.mf:
        plotter = Plotter3D()
        plotter.plot(run_path=Path().cwd())



@hydra.main(config_path="configs", config_name="run", version_base="1.2")
def run(args):
    # Print arguments to stderr (useful on cluster)
    sys.stderr.write(f"{' '.join(sys.argv)}\n")
    sys.stderr.write(f"args = {args}\n\n")
    sys.stderr.flush()

    _set_seeds(args.seed)
    working_directory = Path().cwd()

    # Log general information
    logger.info(f"Using working_directory={working_directory}")
    with contextlib.suppress(TypeError):
        git_info = gitinfo.get_git_info()
        logger.info(f"Commit hash: {git_info['commit']}")
        logger.info(f"Commit date: {git_info['author_date']}")
    logger.info(f"Arguments:\n{OmegaConf.to_yaml(args)}")

    # Actually run
    hydra.utils.call(args.algorithm.run_function, args)
    logger.info("Run finished")
# end of run


def preprocess_tabular(name:str, df: pd.DataFrame) -> pd.DataFrame:
    columns_to_remove = []
    if "lcbench" in name:
        # removing table columns that are not part of the raw search space
        columns_to_remove = [
            'loss',
            'val_accuracy',
            'test_accuracy',
            'test_balanced_accuracy',
            'val_balanced_accuracy',
            'test_cross_entropy',
            'val_cross_entropy',
            'time',
            'imputation_strategy',
            'learning_rate_scheduler',
            'network',
            'normalization_strategy',
            'optimizer',
            'cosine_annealing_T_max',
            'cosine_annealing_eta_min',
            'activation',
            'mlp_shape',
        ]
    elif "pd1" in name:
        # removing table columns that are not part of the raw search space
        columns_to_remove = [
            "train_cost",
            "valid_error_rate",
            "test_error_rate",
            "original_steps",
        ]
    elif "taskset" in name:
        # removing table columns that are not part of the raw search space
        columns_to_remove = [
            'train_loss', 
            'valid1_loss', 
            'valid2_loss', 
            'test_loss', 
            'train_cost',
            'step'
        ]
    elif "nb201" in name:
        columns_to_remove = [
            'train_loss', 
            'train_accuracy', 
            'train_per_time', 
            'train_all_time',
            'test_loss', 
            'test_accuracy', 
            'test_per_time', 
            'test_all_time'
            ]
    _config_ids = df.index.get_level_values('id').unique().values
    # creates a flattened index by retaining only the configuration ID and choosing
    # a placeholder for the fidelity
    # ASSUMPTION: the dataframe is multi-indexed by (id, fidelity)
    df = df.loc[pd.IndexSlice[_config_ids, df.index.values[0][1]], :]
    df = df.set_index(df.index.get_level_values("id"))
    df.index = df.index.astype(int)
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=col, inplace=False)
        else:
            logger.warning(f"Cannot find column `{col}` in table: NOT DROPPED!")
    return df


def set_bounds_from_table(obj, table, space):
    # Set table bounds as SearchSpace bounds here
    if isinstance(space, CS.ConfigurationSpace):
        cs = CS.ConfigurationSpace(space.name)
        hp_name_list = space.get_hyperparameter_names()
        for hp_name in hp_name_list:
            hp = space.get_hyperparameter(name=hp_name)
            hp_class = type(hp)
            lower = table.loc[:, hp_name].min()
            upper = table.loc[:, hp_name].max()
            log = None
            if hasattr(hp, "log"):
                log = hp.log
            # default = hp.default_value
    
            if log is not None:
                cs.add_hyperparameter(
                    hp_class(name=hp_name,lower=lower, upper=upper,log=log)
                )
            else:
                cs.add_hyperparameter(
                    hp_class(name=hp_name, lower=lower, upper=upper)
                )
    
        # both table and space are required to handle tabular spaces
        obj.pipeline_space.set_custom_grid_space(table, cs)
    else:
        for hp_name, hp in space.items():
            if not hp.is_fidelity and hasattr(hp, "lower"):
                lower = table.loc[:, hp_name].min()
                upper = table.loc[:, hp_name].max()
                hp.lower = lower
                hp.upper = upper
                # print(f"HP: {hp_name}\n\tLower: {lower},\n\tUpper: {upper}")
        # both table and space are required to handle tabular spaces
        obj.pipeline_space.set_custom_grid_space(table, space)
    return obj


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

  