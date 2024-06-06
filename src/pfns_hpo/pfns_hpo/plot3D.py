from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path
import multiprocessing as mp
from functools import partial

from argparse import ArgumentParser

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

import itertools

from neps.status.status import get_run_summary_csv
import re
import pandas as pd
import numpy as np

from typing import Callable

# Copied from plot.py
HERE = Path(__file__).parent.absolute()
DEFAULT_RESULTS_PATH = HERE.parent / "results"


# Copied from regret_plot.py
def _find_correct_path(path: Path, strict: bool = False) -> Path:
    if (path / "neps_root_directory").exists():
        path = path / "neps_root_directory"
        assert (path / "summary_csv").exists(), f"Run {path} not completed?"
        path = path / "summary_csv"
        assert (path / "config_data.csv").exists(), f"Run {path} not completed?"
        return path / "config_data.csv"
    if (path / "arlind_root_directory").exists():
        path = path / "arlind_root_directory"
        assert (path / "run_data.csv").exists(), f"Run {path} not completed?"
        return path / "run_data.csv"
    raise ValueError(f"Cannot find relevant subdirectory in {path}!")


def build_a_path(
        experiment_group: str,
        benchmark: str,
        algorithm: str,
        seed: str,
        results_path: str | Path = DEFAULT_RESULTS_PATH,
) -> Path:
    results_path = Path(results_path)

    return results_path / f"{experiment_group}" / f"benchmark={benchmark}" / f"algorithm={algorithm}" / f"seed={seed}"


@dataclass
class Plotter3D:
    algorithm: str | None = None
    benchmark: str | None = None
    seed: int | str | None = None
    experiment_group: str | None = None
    loss: str = "Loss"
    loss_range: tuple[float | int] = (0, 1)
    epochs_range: tuple[int] = (0, 50)
    config_column: str | None = None
    data_path: str | Path | None = None
    base_results_path: str | Path = DEFAULT_RESULTS_PATH
    strict: bool = False
    get_x: Callable[[pd.DataFrame], np.array] | None = None
    get_y: Callable[[pd.DataFrame], np.array] | None = None
    get_z: Callable[[pd.DataFrame], np.array] | None = None
    get_color: Callable[[pd.DataFrame], np.array] | None = None
    scatter: bool = True
    footnote: bool = True
    alpha: float = 0.9
    scatter_size: float | int = 3
    bck_color_2d: tuple[float] = (0.8, 0.82, 0.8)
    view_angle: tuple[float | int] = (15, -70)

    @staticmethod
    def get_x(df: pd.DataFrame) -> np.array:
        return df["epochID"].to_numpy()

    @staticmethod
    def get_y(df: pd.DataFrame) -> np.array:
        y_ = df["configID"].to_numpy()
        return np.ones_like(y_) * y_[0]

    @staticmethod
    def get_z(df: pd.DataFrame) -> np.array:
        return df["result.loss"].to_numpy()

    @staticmethod
    def get_color(df: pd.DataFrame) -> np.array:
        return df.index.to_numpy()

    def get_data_path(self):

        path_2_seed = build_a_path(experiment_group=self.experiment_group,
                                   benchmark=self.benchmark,
                                   algorithm=self.algorithm,
                                   seed=self.seed,
                                   results_path=self.base_results_path)
        try:
            self.data_path = _find_correct_path(path_2_seed)
        except AssertionError as e:
            if self.strict:
                raise Exception(repr(e))
            else:
                get_run_summary_csv(path_2_seed / "neps_root_directory")
                self.data_path = _find_correct_path(path_2_seed)

    def get_info_from_run_path(self, path: str | Path | None = None):

        path = self.data_path if path is None else path

        assert path is not None, "path can't be None"

        match_exp_group = "[\w\d-]+(?=/benchmark)"
        match_benchmark = "(?<=benchmark=)[\w\d-]+(?=/)"
        match_algorithm = "(?<=algorithm=)[\w\d-]+(?=/)"
        match_seed = "(?<=seed=)[\w\d-]+"

        pattern = f"{match_exp_group}|{match_benchmark}|{match_algorithm}|{match_seed}"
        matches = re.findall(pattern, str(Path(path).absolute()))

        try:
            self.experiment_group, self.benchmark, self.algorithm, self.seed = matches
        except ValueError as e:
            raise ValueError(f"Number of matched strings is not equal to 4. matched strings: {matches}") from e

    def prep_df(self, df: pd.DataFrame) -> pd.DataFrame:

        time_cols = ["result.info_dict.start_time", "result.info_dict.end_time"]
        config_columns = ["Config_id", "result.info_dict.config_id"]

        df = df.sort_values(by=time_cols).reset_index(drop=True)

        # Set the config_column attribute according to the DataFrame provided
        if config_columns[0] in df.columns:
            self.config_column = config_columns[0]
        else:
            self.config_column = config_columns[1]

        df[['configID', 'epochID']] = df[self.config_column].str.split('_', expand=True).apply(pd.to_numeric)
        return df

    def plot3D(self,
               data: pd.DataFrame,
               run_path: str | Path | None = None):

        data = self.prep_df(data)

        # Create the figure and the axes for the plot
        fig, (ax3D, ax, cax) = plt.subplots(1, 3, figsize=(12, 5), width_ratios=(20, 20, 1))

        # remove a 2D axis and replace with a 3D projection one
        ax3D.remove()
        ax3D = fig.add_subplot(131, projection='3d')

        # Create the normalizer to normalize the color values
        norm = Normalize(self.get_color(data).min(), self.get_color(data).max())

        # Counters to keep track of the configurations run for only a single fidelity
        n_lines = 0
        n_mins = 0

        data_groups = data.groupby("configID", sort=False)

        for idx, (configID, data_) in enumerate(data_groups):

            x = self.get_x(data_)
            y = self.get_y(data_)
            z = self.get_z(data_)

            y = np.ones_like(y) * idx
            color = self.get_color(data_)

            if len(x) < 2:
                n_mins += 1
                if self.scatter:
                    ax3D.scatter(y, z, s=self.scatter_size, zs=0, zdir="x", c=color, cmap='RdYlBu_r', norm=norm,
                                 alpha=self.alpha * 0.8)
                    ax.scatter(x, z, s=self.scatter_size, c=color, cmap='RdYlBu_r', norm=norm, alpha=self.alpha * 0.8)
            else:
                n_lines += 1

                # Plot 3D
                # Get segments for all lines
                points3D = np.array([x, y, z]).T.reshape(-1, 1, 3)
                segments3D = np.concatenate([points3D[:-1], points3D[1:]], axis=1)

                # Construct lines from segments
                lc3D = Line3DCollection(segments3D, cmap='RdYlBu_r', norm=norm, alpha=self.alpha)
                lc3D.set_array(color)

                # Draw lines
                ax3D.add_collection3d(lc3D)

                # Plot 2D
                # Get segments for all lines
                points = np.array([x, z]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Construct lines from segments
                lc = LineCollection(segments, cmap="RdYlBu_r", norm=norm, alpha=self.alpha)
                lc.set_array(color)

                # Draw lines
                ax.add_collection(lc)

        ax3D.axes.set_xlim3d(left=self.epochs_range[0], right=self.epochs_range[1])
        ax3D.axes.set_ylim3d(bottom=0, top=data_groups.ngroups)
        ax3D.axes.set_zlim3d(bottom=self.loss_range[0], top=self.loss_range[1])

        ax3D.set_xlabel('Epochs')
        ax3D.set_ylabel('Iteration sampled')
        ax3D.set_zlabel(f'{self.loss}')

        # set view angle
        ax3D.view_init(elev=self.view_angle[0], azim=self.view_angle[1])

        ax.autoscale_view()
        ax.set_xlabel('Epochs')
        ax.set_ylabel(f'{self.loss}')
        ax.set_facecolor(self.bck_color_2d)
        fig.suptitle(f"alg: {self.algorithm}, benchmark: {self.benchmark}, seed: {self.seed}")

        if self.footnote:
            fig.text(0.01, 0.02,
                     f'Total {n_lines + n_mins} configs evaluated; for multiple budgets: {n_lines}, for single budget: {n_mins}',
                     ha='left',
                     va="bottom",
                     fontsize=10)

        plt.colorbar(cm.ScalarMappable(norm=norm, cmap="RdYlBu_r"), cax=cax, label='Iteration', use_gridspec=True,
                     alpha=self.alpha)
        fig.tight_layout()

        self.save(run_path)
        plt.close(fig)

    def save(self, run_path: str | Path | None = None):
        if run_path is None:
            run_path = build_a_path(experiment_group=self.experiment_group,
                                    benchmark=self.benchmark,
                                    algorithm=self.algorithm,
                                    seed=self.seed,
                                    results_path=self.base_results_path)
        else:
            run_path = Path(run_path)

        plot_path = run_path / f"Plot3D_{self.experiment_group}_{self.benchmark}_{self.algorithm}_{self.seed}.png"
        plt.savefig(plot_path,
                    bbox_inches='tight')

    def plot(self,
             data: pd.DataFrame | None = None,
             run_path: str | Path | None = None):
        if run_path is None and self.data_path is None and \
        (
            self.algorithm is None or self.benchmark is None or self.experiment_group is None or self.seed is None
        ):
            raise ValueError("At least run_path or self.data_path or all of the arguments: "
                             "self.algorithm, self.benchmark, self.experiment_group, self.seed must be not None")

        elif run_path is None and self.data_path is None:
            self.get_data_path()
        else:
            self.get_info_from_run_path(run_path)
            if self.data_path is None:
                self.get_data_path()

        if data is None:
            data = pd.read_csv(self.data_path, float_precision="round_trip")

        self.plot3D(data, run_path)


# function for multiprocessing.Pool
def plotting_process(benchmark_algorithm_seed,
                     experiment_group,
                     base_dir,
                     loss_name,
                     loss_range,
                     epochs_range,
                     strict,
                     footnote,
                     view_angle):
    # imap accepts only one variable
    benchmark, algorithm, seed = benchmark_algorithm_seed

    try:
        plotter = Plotter3D(algorithm=algorithm,
                            benchmark=benchmark,
                            experiment_group=experiment_group,
                            base_results_path=base_dir,
                            seed=int(seed),
                            loss=loss_name,
                            loss_range=loss_range,
                            epochs_range=epochs_range,
                            strict=strict,
                            footnote=footnote,
                            view_angle=view_angle)
        plotter.plot()
        return f"Run complete: algorithm={algorithm} benchmark={benchmark} seed={seed}"
    except Exception as e:
        return f"Run FAILED: algorithm={algorithm} benchmark={benchmark} seed={seed}" \
               f"\n{repr(e)}"
        # raise RuntimeError(f"An exception occurred during handling: "
        #                    f"\n\talgorithm={algorithm} "
        #                    f"\n\tbenchmark={benchmark} "
        #                    f"\n\tseed={seed}") from e


def main(algorithms: list[str],
         benchmarks: list[str],
         experiment_group: str,
         seeds: list[str | int],
         base_dir: str | Path | None = None,
         strict: bool = False,
         loss_name: str = "Loss",
         loss_range: tuple[float | int] = (0, 1),
         epochs_range: tuple[int] = (0, 50),
         footnote: bool = True,
         view_angle: tuple[float | int] = (15, -70)):
    seeds = range(*seeds)

    # partial for default variables
    plot = partial(plotting_process,
                   experiment_group=experiment_group,
                   base_dir=base_dir,
                   loss_name=loss_name,
                   loss_range=loss_range,
                   epochs_range=epochs_range,
                   strict=strict,
                   footnote=footnote,
                   view_angle=view_angle)

    # Use mp.Pool here instead of joblib.Parallel to avoid extra overhead
    # Since each job runs for about 3seconds
    # see https://stackoverflow.com/a/57710710/8889365
    with mp.Pool(processes=None) as p:
        for result in p.imap(plot, itertools.product(benchmarks, algorithms, seeds)):
            print(result)


def parse_args():
    parser = ArgumentParser(description="3D plotting")

    parser.add_argument("--basedir", type=str, default=None)
    parser.add_argument("--expgroup", type=str, default=None, required=True)

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Expects a list of algorithms, comma separated str. Needs at least one."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Expects a list of benchmarks, comma separated str. Needs at least one."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        required=False,
        help=(
            "If len(args.seeds) is 1, then interprets it as `range(args.seeds)`. "
            "If len(args.seeds) is 2, then interprets it as `range(*args.seeds)`. "
            "Anything else will throw an error."
        )
    )

    parser.add_argument("--epochs_range", nargs=2, type=float, default=[0, 50])
    parser.add_argument("--loss_range", nargs=2, type=float, default=[0, 1], required=False)
    parser.add_argument("--view_angle", nargs=2, type=float, default=[15, -70], required=False)

    parser.add_argument(
        "--strict",
        action="store_true",
        help="If True, requires that no runs are missing."
    )

    parser.add_argument(
        "--footnote",
        action="store_true",
        help="If True, add footnote counting single fidelity configurations."
    )

    parser.add_argument("--loss_name", type=str, default="Loss", required=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert args.seeds is not None and len(args.seeds) <= 2, "Invalid --seeds. Check --help."
    assert len(args.algorithms) > 0, "Invalid --algorithms. Check --help."
    assert len(args.benchmarks) > 0, "Invalid --benchmarks. Check --help."
    assert args.loss_range is not None and len(args.loss_range) == 2, "Invalid --loss_range. Check --help."
    assert args.epochs_range is not None and len(args.epochs_range) == 2, "Invalid --epochs_range. Check --help."
    assert args.view_angle is not None and len(args.view_angle) == 2, "Invalid --view_angle. Check --help."

    args.basedir = DEFAULT_RESULTS_PATH if args.basedir is None else Path(args.basedir)
    assert args.basedir.exists(), f"Base path: {args.basedir} does not exist!"

    main(
        algorithms=args.algorithms,
        benchmarks=args.benchmarks,
        experiment_group=args.expgroup,
        seeds=args.seeds,
        base_dir=args.basedir,
        loss_name=args.loss_name,
        strict=args.strict,
        loss_range=args.loss_range,
        epochs_range=args.epochs_range,
        footnote=args.footnote,
        view_angle=args.view_angle)
