import argparse
import os
import time

import numpy as np
import pfns4hpo
import torch
from benchmark import get_benchmark
from taskids import lcbench_ids, pd1_ids, taskset_ids


def evaluate(benchmark_name, task_id, sep, cases_per_dataset, model_name, data_path):
    benchmark, output_name = get_benchmark(benchmark_name, task_id, data_path)

    data = torch.load(
        os.path.join(
            "tasks", benchmark, f"{output_name}_{cases_per_dataset}bs_{sep}sep.pt"
        )
    )
    data = torch.swapaxes(data, 0, 1).float()
    data[..., 1] = data[..., 1] / benchmark.end
    train_x = data[:sep, :, :-1]
    train_y = 1 - data[:sep, :, -1]
    test_x = data[sep:, :, :-1]
    test_y = 1 - data[sep:, :, -1]

    model = pfns4hpo.PFN_MODEL(name=model_name)

    results = []
    for i in range(data.shape[1]):
        start = time.time()
        mask_test = torch.where(test_x[:, i].sum(-1) != 0)[0].long()
        res = -model.nll_loss(
            x_train=train_x[:, i],
            y_train=train_y[:, i],
            x_test=test_x[mask_test, i],
            y_test=test_y[mask_test, i],
        )
        ll = res.mean().item()
        end = time.time()
        ypred = model.predict_mean(
            x_train=train_x[:, i], y_train=train_y[:, i], x_test=test_x[mask_test, i]
        )
        mse = ((ypred.flatten() - test_y[mask_test, i]) ** 2).mean().item()

        results.append([ll, mse, end - start])

    results = np.array(results)
    np.save(
        os.path.join(
            "results",
            benchmark_name,
            f"{output_name}_{cases_per_dataset}bs_{sep}sep_{model_name}.npy",
        ),
        results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pfn", help="model to evaluate")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="lcbench_tabular",
        help="benchmark to evaluate on",
    )
    parser.add_argument(
        "--data_path", type=str, default="../../../data/", help="Path to data"
    )
    args = parser.parse_args()
    os.makedirs(os.path.join("results", args.benchmark), exist_ok=True)

    benchmark_map = {
        "lcbench_tabular": lcbench_ids,
        "pd1_tabular": pd1_ids,
        "taskset_tabular": taskset_ids,
    }

    configs = [
        {
            "benchmark_name": args.benchmark,
            "task_id": tid,
            "sep": sep,
            "cases_per_dataset": args.cases_per_dataset,
            "model_name": args.model,
            "data_path": args.data_path,
        }
        for sep in [100, 200, 400, 600, 800, 1000, 1200, 1400, 1800]
        for tid in benchmark_map.get(args.benchmark, [])
    ]

    for config in configs:
        evaluate(**config)
