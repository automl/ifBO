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
    nb_configs = len(benchmark.configs)
    original_id = np.arange(nb_configs)
    offset = min([int(_) for _ in benchmark.configs.keys()])

    x = []
    y = []

    for i in range(ntasks_per_dataset):
        epoch = np.zeros(seq_len)
        id_curve = np.zeros(seq_len)
        curve_val = np.zeros(seq_len)

        ok = False

        while not ok:
            n_levels = int(np.round(10 ** np.random.uniform(0, 3)))
            n_levels = min(n_levels, max_fidelities)

            alpha = 10 ** np.random.uniform(-4, -1)
            weights = np.random.gamma(alpha, alpha, min(1000, nb_configs)) + EPS
            p = weights / np.sum(weights)
            ids = np.arange(min(1000, nb_configs))
            all_levels = np.repeat(ids, n_levels)
            all_p = np.repeat(p, n_levels) / n_levels

            if len(all_levels) > seq_len:
                ok = True

        ordering = np.random.choice(all_levels, p=all_p, size=seq_len, replace=False)

        cutoff_per_curve = np.zeros((seq_len,), dtype=int)
        epochs_per_curve = np.zeros((seq_len,), dtype=int)
        for i in range(seq_len):  # loop over every pos
            cid = ordering[i]
            epochs_per_curve[cid] += 1
            if i < single_eval_pos:
                cutoff_per_curve[cid] += 1

        c_minus_a = np.random.uniform()  # epoch 0

        # determine config, x, y for every curve
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
            id_curve[i] = cid + 1  # reserve ID 0 for queries

            epoch[i] = curve_xs[cid][curve_counters[cid]]
            curve_counters[cid] += 1

        uid = np.unique(id_curve[:single_eval_pos])

        nbiud = len(id_curve[single_eval_pos : (single_eval_pos + len(uid))])
        id_curve[single_eval_pos : (single_eval_pos + len(uid))] = uid[:nbiud]
        epoch[single_eval_pos : min(single_eval_pos + len(uid), 2000)] = max_fidelities

        x.append([id_curve, epoch])

    complete_data = []
    for id_curve, epoch in x:
        np.random.shuffle(original_id)
        final_data = []
        for ordering, config_id, fidelity in zip(
            id_curve, original_id[id_curve.astype(int) - 1], epoch
        ):
            if ordering == 0:
                tmp = [0] * len(final_data[-1])
            else:
                _config_id = str(config_id + offset)
                tmp = []
                tmp = tmp + [ordering, fidelity]
                tmp = tmp + _get_normalized_values(
                    config=benchmark.configs[_config_id], configuration_space=space
                )
                tmp = tmp + [benchmark.query(config=_config_id, at=fidelity).error]
            final_data.append(tmp)
        complete_data.append(final_data)
    complete_data = np.array(complete_data).astype(np.float32)
    complete_data = torch.from_numpy(complete_data)

    torch.save(
        complete_data,
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
