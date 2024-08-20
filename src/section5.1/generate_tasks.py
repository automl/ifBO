import os
import torch
import argparse
import numpy as np
from pfns4hpo.evaluate import _get_normalized_values
from taskids import lcbench_ids, pd1_ids, taskset_ids
from benchmark import get_benchmark


def generate_tasks(
    benchmark_name, task_id, ntasks_per_dataset, single_eval_pos, data_path, seq_len
):
    EPS = 10**-9
    benchmark, output_name = get_benchmark(benchmark_name, task_id, data_path)

    space = benchmark.space
    max_fidelities = benchmark.end
    ncurves = len(benchmark.configs)
    original_id = np.arange(ncurves)
    offset = min([int(_) for _ in benchmark.configs.keys()])

    allocations = []

    for i in range(ntasks_per_dataset):
        epoch = np.zeros(seq_len)
        id_curve = np.zeros(seq_len)

        ok = False

        # determine # observations/queries per curve
        while not ok:
            n_levels = int(np.round(10 ** np.random.uniform(0, 3)))
            n_levels = min(n_levels, max_fidelities)

            alpha = 10 ** np.random.uniform(-4, -1)
            weights = np.random.gamma(alpha, alpha, min(1000, ncurves)) + EPS
            p = weights / np.sum(weights)
            ids = np.arange(min(1000, ncurves))
            all_levels = np.repeat(ids, n_levels)
            all_p = np.repeat(p, n_levels) / n_levels
            if len(all_levels) > seq_len:
                ok = True
        ordering = np.random.choice(all_levels, p=all_p, size=seq_len, replace=False)

        # calculate the cutoff/samples for each curve
        cutoff_per_curve = np.zeros((seq_len,), dtype=int)
        epochs_per_curve = np.zeros((seq_len,), dtype=int)
        for i in range(seq_len):  # loop over every pos
            cid = ordering[i]
            epochs_per_curve[cid] += 1
            if i < single_eval_pos:
                cutoff_per_curve[cid] += 1

        # determine config, epochs for every curve
        curve_xs = []
        for cid in range(seq_len):  # loop over every curve
            if epochs_per_curve[cid] > 0:
                x_ = np.zeros((epochs_per_curve[cid],))
                if cutoff_per_curve[cid] > 0:  # observations (if any)
                    x_[: cutoff_per_curve[cid]] = np.arange(
                        1, cutoff_per_curve[cid] + 1
                    )
                if cutoff_per_curve[cid] < epochs_per_curve[cid]:  # queries (if any)
                    x_[cutoff_per_curve[cid] :] = np.random.choice(
                        np.arange(cutoff_per_curve[cid] + 1, n_levels + 1),
                        size=epochs_per_curve[cid] - cutoff_per_curve[cid],
                        replace=False,
                    )
                curve_xs.append(x_)
            else:
                curve_xs.append(None)

        # construct the batch data element
        curve_counters = np.zeros(seq_len, dtype=np.int64)
        for i in range(single_eval_pos):
            cid = ordering[i]
            id_curve[i] = cid + 1  # start from 1
            epoch[i] = curve_xs[cid][curve_counters[cid]]
            curve_counters[cid] += 1

        # assign max fidelity to all curves in context
        # specific to the evaluation in 5.1
        unique_curves = np.unique(id_curve[:single_eval_pos])
        num_unique_curves = len(unique_curves)
        id_curve[single_eval_pos : single_eval_pos + num_unique_curves] = unique_curves
        end_pos = min(single_eval_pos + num_unique_curves, seq_len)
        epoch[single_eval_pos:end_pos] = max_fidelities

        allocations.append([id_curve, epoch])

    all_tasks = []
    for id_curve, epoch in allocations:
        np.random.shuffle(original_id)
        task_data = []
        for ordering, config_id, fidelity in zip(
            id_curve, original_id[id_curve.astype(int) - 1], epoch
        ):
            if ordering == 0:
                tmp = [0] * len(task_data[-1])
            else:
                _config_id = str(config_id + offset)
                tmp = []
                tmp = tmp + [ordering, fidelity]
                tmp = tmp + _get_normalized_values(
                    config=benchmark.configs[_config_id], configuration_space=space
                )
                tmp = tmp + [benchmark.query(config=_config_id, at=fidelity).error]
            task_data.append(tmp)
        all_tasks.append(task_data)
    all_tasks = np.array(all_tasks).astype(np.float32)
    all_tasks = torch.from_numpy(all_tasks)

    torch.save(
        all_tasks,
        os.path.join(
            "tasks",
            benchmark_name,
            f"{output_name}_{ntasks_per_dataset}bs_{single_eval_pos}sep.pt",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntasks_per_dataset",
        type=int,
        default=100,
        help="Number of cases to generate per dataset.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="lcbench_tabular",
        help="Benchmark to generate cases for.",
    )
    parser.add_argument("--seed", type=int, help="Seed for random number generator.")
    parser.add_argument(
        "--data_path", type=str, default="../../../data/", help="Path to data"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(os.path.join("tasks", args.benchmark), exist_ok=True)

    benchmark_map = {
        "lcbench_tabular": lcbench_ids,
        "pd1_tabular": pd1_ids,
        "taskset_tabular": taskset_ids,
    }

    configs = [
        {
            "benchmark": args.benchmark,
            "task_id": tid,
            "ntasks_per_dataset": args.ntasks_per_dataset,
            "single_eval_pos": nsamples,
            "data_path": args.data_path,
            "seq_len": 2000,
        }
        for nsamples in [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
        for tid in benchmark_map.get(args.benchmark, [])
    ]

    for config in configs:
        generate_tasks(**config)
