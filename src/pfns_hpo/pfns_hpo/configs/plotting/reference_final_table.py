import argparse
import errno
import os
import time
from multiprocessing import Manager

import numpy as np
import pandas as pd
from attrdict import AttrDict
from joblib import Parallel, delayed, parallel_backend
from path import Path
from scipy import stats

from .configs.plotting.read_results import get_seed_info, SINGLE_FIDELITY_ALGORITHMS
from .configs.plotting.utils import interpolate_time

benchmark_configs_path = os.path.join(os.path.dirname(__file__), "configs/benchmark/")


def _process_seed(
    _path,
    seed,
    algorithm,
    key_to_extract,
    cost_as_runtime,
    results,
    n_workers,
    parallel_sleep_decrement,
):
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime())}] "
        f"[-] [{algorithm}] Processing seed {seed}..."
    )

    # `algorithm` is passed to calculate continuation costs
    losses, infos, max_cost = get_seed_info(
        _path,
        seed,
        algorithm=algorithm,
        cost_as_runtime=cost_as_runtime,
        n_workers=n_workers,
        parallel_sleep_decrement=parallel_sleep_decrement
    )
    incumbent = np.minimum.accumulate(losses)
    cost = [i[key_to_extract] for i in infos]
    results["incumbents"].append(incumbent)
    results["costs"].append(cost)
    results["max_costs"].append(max_cost)


def plot(args):

    starttime = time.time()

    BASE_PATH = (
        Path(__file__).parent / "../.."
        if args.base_path is None
        else Path(args.base_path)
    )

    KEY_TO_EXTRACT = "cost" if args.cost_as_runtime else "fidelity"

    base_path = BASE_PATH / "results" / args.experiment_group
    output_dir = BASE_PATH / "tables" / args.experiment_group

    print(
        f"[{time.strftime('%H:%M:%S', time.localtime())}]"
        f" Processing {len(args.benchmarks)} benchmarks "
        f"and {len(args.algorithms)} algorithms..."
    )

    final_table = dict()
    for benchmark_idx, benchmark in enumerate(args.benchmarks):
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] "
            f"[{benchmark_idx}] Processing {benchmark} benchmark..."
        )
        benchmark_starttime = time.time()

        _base_path = os.path.join(base_path, f"benchmark={benchmark}")
        if not os.path.isdir(_base_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _base_path)
        for algorithm in args.algorithms:
            _path = os.path.join(_base_path, f"algorithm={algorithm}")
            if not os.path.isdir(_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _path)

            algorithm_starttime = time.time()
            seeds = sorted(os.listdir(_path))

            if args.parallel:
                manager = Manager()
                results = manager.dict(
                    incumbents=manager.list(),
                    costs=manager.list(),
                    max_costs=manager.list(),
                )
                with parallel_backend(args.parallel_backend, n_jobs=-1):
                    Parallel()(
                        delayed(_process_seed)(
                            _path,
                            seed,
                            algorithm,
                            KEY_TO_EXTRACT,
                            args.cost_as_runtime,
                            results,
                            args.n_workers,
                            args.parallel_sleep_decrement,
                        )
                        for seed in seeds
                    )

            else:
                results = dict(incumbents=[], costs=[], max_costs=[])
                # pylint: disable=expression-not-assigned
                [
                    _process_seed(
                        _path,
                        seed,
                        algorithm,
                        KEY_TO_EXTRACT,
                        args.cost_as_runtime,
                        results,
                        args.parallel_sleep_decrement,
                    )
                    for seed in seeds
                ]

            incumbents = np.array(results["incumbents"][:])
            costs = np.array(results["costs"][:])
            max_cost = None if args.cost_as_runtime else max(results["max_costs"][:])

            df = interpolate_time(
                incumbents,
                costs,
                scale_x=max_cost,
                parallel_evaluation=(args.n_workers > 1),
                rounded_integer_costs_for_x_range=(algorithm in SINGLE_FIDELITY_ALGORITHMS)
            )

            if args.budget is not None:
                df = df.query(f"index <= {args.budget}")
            final_mean = df.mean(axis=1).values[-1]
            final_std_error = stats.sem(df.values, axis=1)[-1]

            if benchmark not in final_table:
                final_table[benchmark] = dict()
            final_table[benchmark][
                algorithm
            ] = rf"${np.round(final_mean, 2)} \pm {np.round(final_std_error, 2)}$"

            print(f"Time to process algorithm data: {time.time() - algorithm_starttime}")
        print(f"Time to process benchmark data: {time.time() - benchmark_starttime}")

    final_table = pd.DataFrame.from_dict(final_table, orient="index")

    filename = args.filename
    if filename is None:
        filename = f"{args.experiment_group}_{args.plot_id}"

    output_dir = Path(output_dir)
    output_dir.makedirs_p()

    with open(
        os.path.join(output_dir, f"{filename}.tex"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\\begin{table}[htbp]" + " \n")
        f.write("\\centering" + " \n")

        f.write(
            "\\begin{tabular}{"
            + " | ".join(["c"] * (len(final_table.columns) + 1))
            + "}\n"
        )
        f.write("\\toprule" + " \n")
        f.write("{} ")

        import re

        for c in final_table.columns:
            f.write("& ")
            f.write(re.sub("_", r"\\_", c) + " ")
        f.write("\\\\\n")
        f.write("\\midrule" + " \n")

        for _, row in final_table.iterrows():
            f.write(re.sub("_", r"\\_", str(row.name)) + " ")
            f.write(" & " + " & ".join([str(x) for x in row.values]))
            f.write(" \\\\\n")
        f.write("\\bottomrule" + " \n")
        f.write("\\end{tabular}" + " \n")
        budget_caption = (
            ""
            if args.budget is None
            else f" until {args.budget} {'full function evaluations' if not args.cost_as_runtime else 's'}"
        )
        f.write("\\caption{" f"{args.caption + budget_caption}" + "}" + " \n")
        f.write("\\label{" f"{args.label}" + "}" + " \n")
        f.write("\\end{table}")
        # f.write(final_table.to_latex())
    print(f"{final_table}")
    print(f'Saved to "{output_dir}/{filename}.tex"')
    print(f"Processing took {time.time() - starttime}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="mf-prior-exp plotting",
    )
    parser.add_argument(
        "--base_path", type=str, default=None, help="path where `results/` exists"
    )
    parser.add_argument("--experiment_group", type=str, default="")
    parser.add_argument("--caption", type=str, default="TODO")
    parser.add_argument("--label", type=str, default="TODO")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--plot_id", type=str, default="1")
    parser.add_argument(
        "--filename", type=str, default=None, help="name out pdf file generated"
    )
    parser.add_argument(
        "--cost_as_runtime",
        default=False,
        action="store_true",
        help="Default behaviour to use fidelities on the x-axis. "
        "This parameter uses the training cost/runtime on the x-axis",
    )
    parser.add_argument(
        "--parallel",
        default=False,
        action="store_true",
        help="whether to process data in parallel or not",
    )
    parser.add_argument(
        "--parallel_backend",
        type=str,
        choices=["multiprocessing", "threading"],
        default="multiprocessing",
        help="which backend use for parallel",
    )
    args = AttrDict(parser.parse_args().__dict__)
    plot(args)  # pylint: disable=no-value-for-parameter
