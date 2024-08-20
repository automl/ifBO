import torch
import mfpbench
import neps
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

def _get_normalized_values(config, configuration_space: ConfigurationSpace) -> list[float]:
    """Get normalized values of a given configuration."""
    list_hp_names = configuration_space.get_hyperparameter_names()
    dict_values = config.as_dict()
    dict_values = dict((hp, dict_values[hp]) for hp in list_hp_names)

    neps_cfg = neps.search_spaces.search_space.SearchSpace(
        **{
            k: v 
            for k, v 
            in neps.search_spaces.search_space.pipeline_space_from_configspace(configuration_space).items()
            if k in list_hp_names
           }
    )

    neps_cfg.set_hyperparameters_from_dict(dict_values, defaults=False)

    res = [neps_cfg[k].normalized().value for k in list_hp_names]
    if any([res[_] is None for _ in range(len(res))]):
        print("WARNING: NaN values found in normalized values.")
        import pudb; pudb.set_trace()
    
    return res

def prepare_data(benchmark="tabular-lcbench", task_id="adult", scenario={"train": {}, "test": {}}):
    # verify scenario has train and test keys
    assert "train" in scenario.keys(), "scenario must have a train key"
    assert "test" in scenario.keys(), "scenario must have a test key"

    # check train and test scenario is not empty
    assert len(scenario["train"]) > 0, "train scenario is empty"

    assert benchmark == "tabular-lcbench", "Only lcbench is supported for now."

    print("loading benchmark...")
    benchmark = mfpbench.lcbench_tabular.LCBenchTabularBenchmark(task_id=task_id, 
                                                                datadir="/work/dlclarge1/mallik-lcpfn-hpo/pfns_hpo/data/lcbench-tabular")
    configspace = benchmark.space

    train_data = torch.FloatTensor(
        sum([
                [
                    [config_id, fidelity] 
                    
                        + _get_normalized_values(config=benchmark.configs[str(config_id)], configuration_space=configspace) 
                        + [benchmark.query(config=config_id, at=fidelity).score]
                    for fidelity in fidelities
                ] 
                for config_id, fidelities in scenario["train"].items()], 
            [])
        )
    
    if len(scenario["test"]) == 0:
        return train_data[:, :-1], train_data[:, -1], None, None

    test_data = torch.FloatTensor(
        sum([
                [
                    [config_id, fidelity] 
                        + _get_normalized_values(config=benchmark.configs[str(config_id)], configuration_space=configspace) 
                        + [benchmark.query(config=config_id, at=fidelity).score]
                    for fidelity in fidelities
                ] 
                for config_id, fidelities in scenario["test"].items()], 
            [])
        )
    return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

def compute_loglikelihood(model, benchmark="lcbench", task_id="adult", scenario={"train": {}, "test": {}}):
    """Compute loglikelihood of a given scenario on a given benchmark."""

    x_train, y_train, x_test, y_test = prepare_data(benchmark=benchmark, task_id=task_id, scenario=scenario)
    
    print("computing loglikelihood...")
    return model.nll_loss(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

def get_partial_curves(benchmark="lcbench", n=10):
    """Get partial curves of a given benchmark."""
    assert benchmark != "lcbench", "Only lcbench is supported for now."

    print("loading benchmark...")
    benchmark = mfpbench.lcbench_tabular.LCBenchTabularBenchmark(task_id="adult", 
                                                                datadir="/work/dlclarge1/mallik-lcpfn-hpo/pfns_hpo/data/lcbench-tabular")
    configs = benchmark.sample(n=n)
    curves = [[_.score for _ in benchmark.trajectory(config=config)] for config in configs]
    random_fidelities = np.random.randint(low=benchmark.start, high=benchmark.end, size=n)
    return [curve[:fidelity] for curve, fidelity in zip(curves, random_fidelities)]

def random_selection(benchmark="tabular-lcbench", task_id="adult", max_iteration=1000, return_schedule=False):
    """Random selection of configurations
    Returns a list of observed scores
    """

    assert benchmark == "tabular-lcbench", "Only tabular-lcbench is supported for now."

    benchmark = mfpbench.lcbench_tabular.LCBenchTabularBenchmark(task_id=task_id, datadir="/work/dlclarge1/mallik-lcpfn-hpo/pfns_hpo/data/lcbench-tabular")
    configurations = [_get_normalized_values(config=config, configuration_space=benchmark.space) for config in benchmark.configs.values()]

    assert len(configurations) == 2000

    epochs_per_configuration = torch.zeros(2000)
    trajectory = []
    schedule = []
    curve_id = {}
    idx_curve = 1

    for _ in range(max_iteration):
        selected_configuration = torch.randint(low=0, high=2000, size=(1,)).item()
        if selected_configuration not in curve_id: 
            curve_id[selected_configuration] = idx_curve
            idx_curve += 1
        
        epochs_per_configuration[selected_configuration] = epochs_per_configuration[selected_configuration] + 1
        score = benchmark.query(config=benchmark.configs[str(selected_configuration)], at=epochs_per_configuration[selected_configuration].item()).score

        trajectory.append(score)
        schedule.append(selected_configuration)

    if return_schedule:
        return trajectory, schedule
    else:
        return trajectory



def dynamic_selection(model, benchmark="tabular-lcbench", task_id="adult", max_iteration=1000, criterion="ucb", at="max", return_schedule=False):
    """Sequential selection of configurations using UCB acquisition function at max budget
    Returns a list of observed scores
    """

    assert benchmark == "tabular-lcbench", "Only tabular-lcbench is supported for now."
    assert criterion in ["ucb", "ei"], "Only ucb and ei are supported for now."

    benchmark = mfpbench.lcbench_tabular.LCBenchTabularBenchmark(task_id=task_id, datadir="/work/dlclarge1/mallik-lcpfn-hpo/pfns_hpo/data/lcbench-tabular")
    configurations = [_get_normalized_values(config=config, configuration_space=benchmark.space) for config in benchmark.configs.values()]

    assert len(configurations) == 2000

    x_train = torch.FloatTensor([])
    y_train = torch.FloatTensor([])

    x_test = torch.FloatTensor([
        [0, 1] + config 
        for config in configurations
    ]).float()

    epochs_per_configuration = torch.zeros(2000)
    trajectory = []
    schedule = []
    curve_id = {}
    idx_curve = 1

    for _ in range(max_iteration):
        if criterion == "ucb":
            utility = model.get_ucb(x_train=x_train, y_train=y_train, x_test=x_test).flatten()
        else:
            f_best = torch.FloatTensor([max(trajectory) if len(trajectory) > 0 else 0]).expand(x_test.shape[0]).unsqueeze(-1)
            utility = model.get_ei(x_train=x_train, y_train=y_train, x_test=x_test, f_best=f_best).flatten()
        utility[epochs_per_configuration >= benchmark.end] = -np.inf # set to -inf the configurations that have reached the max budget

        selected_configuration = utility.argmax()
        if selected_configuration not in curve_id: 
            curve_id[selected_configuration] = idx_curve
            idx_curve += 1
        
        epochs_per_configuration[selected_configuration] = epochs_per_configuration[selected_configuration] + 1
        score = benchmark.query(config=benchmark.configs[str(selected_configuration.item())], at=epochs_per_configuration[selected_configuration].item()).score

        x_train = torch.cat([x_train, x_test[selected_configuration:selected_configuration+1]], dim=0)
        x_train[-1, 0] = curve_id[selected_configuration] 
        x_train[-1, 1] = epochs_per_configuration[selected_configuration] / benchmark.end
        y_train = torch.cat([y_train, torch.tensor([score])], dim=0)
        x_test[selected_configuration, 0] = curve_id[selected_configuration]

        trajectory.append(score)
        schedule.append(selected_configuration.item())

    if return_schedule:
        return trajectory, schedule
    else:
        return trajectory

