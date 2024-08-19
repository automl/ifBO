import os

import mfpbench

from pfns_hpo.run import process_taskset_mfpbench_with_step_0_prior


def get_benchmark(name, task_id, data_path):
    if name == "lcbench_tabular":
        datadir = os.path.join(data_path, "lcbench-tabular")
        benchmark = mfpbench.get(
            name=name,
            task_id=task_id,
            datadir=datadir,
            preload=True,
            prior=None,
            remove_constants=True,
            seed=True,
            value_metric="val_balanced_accuracy",
            value_metric_test="test_balanced_accuracy",
        )
    elif name == "pd1_tabular":
        output_name = f"{task_id['model']}_{task_id['dataset']}_{task_id['batch_size']}"
        if "coarseness" in task_id:
            output_name = f"{output_name}_{task_id['coarseness']}"
        datadir = os.path.join(data_path, "pd1-tabular")
        benchmark = mfpbench.get(
            name=name,
            datadir=datadir,
            preload=True,
            prior=None,
            remove_constants=True,
            seed=True,
            **task_id,
        )
    elif name == "taskset_tabular":
        datadir = os.path.join(data_path, "taskset-tabular")
        output_name = f"{task_id['task_id']}_{task_id['optimizer']}"
        benchmark = mfpbench.get(
            name=name,
            datadir=datadir,
            preload=True,
            prior=None,
            seed=True,
            **task_id,
        )
        benchmark = process_taskset_mfpbench_with_step_0_prior(
            benchmark=benchmark, drop_step_0=True
        )
    return benchmark, output_name
