import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from benchmark import get_benchmark
from scipy.stats import norm
from taskids import lcbench_ids, pd1_ids, taskset_ids


def process_hpo_data(hpo_matrix):
    unique_id = torch.unique(hpo_matrix[:, 0])

    configs, learning_curves, epochs = [], [], []

    for idx in unique_id:
        config_indices = torch.where(hpo_matrix[:, 0] == idx)[0]

        configs.append(hpo_matrix[config_indices[0]][2:-1])
        learning_curves.append(hpo_matrix[config_indices][:, -1])
        epochs.append(hpo_matrix[config_indices][:, 1])

        assert len(learning_curves[-1]) == len(epochs[-1])

    return torch.stack(configs, dim=0), learning_curves, epochs, unique_id


def prepare_training_data(configs, learning_curves, budgets):
    lcfgs, llcs, lbgs, ly = [], [], [], []
    for idx, c in enumerate(configs):
        for i in range(len(learning_curves[idx])):
            lcfgs.append(c)
            llcs.append(learning_curves[idx][:i])
            lbgs.append(budgets[idx][i])
            ly.append(learning_curves[idx][i])
    llcs.append(torch.FloatTensor([0, 0, 0.0]))
    return (
        torch.stack(lcfgs),
        torch.nn.utils.rnn.pad_sequence(llcs, batch_first=True)[:-1],
        torch.stack(lbgs, dim=0),
        torch.stack(ly, dim=0),
    )


def prepare_test_data(data, learning_curves, uid):
    lcfgs, llcs, lbgs, ly = [], [], [], []

    for idx, tid in enumerate(data[:, 0]):
        if tid in uid:
            id_in_train = torch.where(uid == tid)[0][0]
            lcfgs.append(data[idx][2:-1])
            llcs.append(learning_curves[id_in_train])
            lbgs.append(data[idx][1])
            ly.append(data[idx][-1])
        else:
            lcfgs.append(data[idx][2:-1])
            llcs.append(torch.FloatTensor([]))
            lbgs.append(data[idx][1])
            ly.append(data[idx][-1])
    llcs.append(torch.FloatTensor([0, 0, 0.0]))
    return (
        torch.stack(lcfgs),
        torch.nn.utils.rnn.pad_sequence(llcs, batch_first=True)[:-1],
        torch.stack(lbgs, dim=0),
        torch.stack(ly, dim=0),
    )


def evaluate(benchmark_name, task_id, sep, cases_per_dataset, model_name, data_path):
    import sys

    sys.path.append(str(Path(__file__).parent.parent.absolute() / "DPL"))
    from surrogate_models.power_law_surrogate import PowerLawSurrogate

    benchmark, output_name = get_benchmark(benchmark_name, task_id, data_path)

    data = torch.load(
        os.path.join(
            "tasks", benchmark_name, f"{output_name}_{cases_per_dataset}bs_{sep}sep.pt"
        )
    )
    data[..., -1] = 1 - data[..., -1]

    results = []
    for i in range(data.shape[0]):
        X = data[i]
        mask = torch.where(X[..., :-1].sum(-1) != 0)[0].long()
        X = X[mask, ...]

        configurations, learning_curves, budgets, unique_ids = process_hpo_data(X[:sep])
        train_x, train_lc, train_budgets, train_y = prepare_training_data(
            configurations, learning_curves, budgets
        )
        test_x, test_lc, test_budgets, test_y = prepare_test_data(
            X[sep:], learning_curves, unique_ids
        )

        input = {
            "X_train": train_x.float(),
            "train_budgets": train_budgets.float() / benchmark.end,
            "train_curves": train_lc.float(),
            "y_train": train_y.float(),
        }
        input_test = {
            "X_test": test_x.float(),
            "test_budgets": test_budgets.float() / benchmark.end,
            "test_curves": test_lc.float(),
            "y_test": test_y.float(),
        }

        model = PowerLawSurrogate(
            np.zeros((1, data.shape[-1] - 3)),
            seed=1,
            max_benchmark_epochs=benchmark.end,
            ensemble_size=5,
            nr_epochs=250,
            fantasize_step=1,
            minimization=True,
            total_budget=1000,
            device="cpu",
            dataset_name="no_name",
            output_path="/tmp",
            max_value=0,
            min_value=1,
            result_dir="",
        )

        start = time.time()

        model.util_train_surrogate(
            train_examples=input["X_train"],
            train_labels=input["y_train"].cpu().numpy(),
            train_budgets=input["train_budgets"],
            train_curves=input["train_curves"],
        )
        real_budgets = (test_lc != 0).sum(dim=1)
        predictions = model.util_predict(
            configurations=input_test["X_test"],
            budgets=input_test["test_budgets"],
            network_real_budgets=real_budgets,
            hp_curves=input_test["test_curves"],
        )

        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        ll = norm.logpdf(test_y.cpu().numpy(), loc=mean, scale=std).mean()

        end = time.time()
        mse = ((mean - test_y.cpu().numpy()) ** 2).mean()
        results.append([ll.item(), mse, end - start])

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
            "model_name": "dpl",
            "data_path": args.data_path,
        }
        for sep in [100, 200, 400, 600, 800, 1000, 1200, 1400, 1800]
        for tid in benchmark_map.get(args.benchmark, [])
    ]

    for c in configs:
        evaluate(**c)
