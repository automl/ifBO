from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import mfpbench
from mfpbench import Benchmark

SEED = 1
N_SAMPLES = 25
EPSILON = 1e-3
MAX_ITERATIONS = 5_000

STYLES: dict = {}


class RunningStats:
    # https://stackoverflow.com/a/17637351

    def __init__(self) -> None:  # noqa: D107
        self.n = 0
        self.old_m = np.array(0)
        self.new_m = np.array(0)
        self.old_s = np.array(0)
        self.new_s = np.array(0)
        self.previous_m = np.array(0)
        self.previous_s = np.array(0)

    def clear(self) -> None:
        """Clear the running stats."""
        self.n = 0

    def push(self, x: np.ndarray) -> None:
        """Push a new value into the running stats."""
        self.n += 1
        self.previous_m = self.old_m
        self.previous_s = self.old_s

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.array(0)
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self) -> np.ndarray:
        """Return the mean of the running stats."""
        return self.new_m if self.n > 1 else 0.0  # type: ignore

    def variance(self) -> np.ndarray:
        """Return the variance of the running stats."""
        return self.new_s / (self.n - 1) if self.n > 1 else np.array(0.0)

    def std(self) -> np.ndarray:
        """Return the standard deviation of the running stats."""
        return np.asarray(np.sqrt(self.variance()))


def correlation_curve(
    b: Benchmark,
    *,
    n_samples: int = 25,
    method: Literal["spearman", "kendalltau", "cosine"] = "spearman",
) -> np.ndarray:
    """Compute the correlation curve for a benchmark.

    Args:
        b: The benchmark to compute the correlation curve for
        n_samples: The number of samples to take from the benchmark
        method: The method to use for computing the correlation curve

    Returns:
        The mean correlation curve
    """
    configs = b.sample(n_samples)
    frame = b.frame()
    for config in configs:
        trajectory = b.trajectory(config)
        for r in trajectory:
            frame.add(r)

    correlations = frame.correlations(method=method)
    return correlations[-1, :]


def plot(  # noqa: D103, C901, PLR0913, PLR0912, PLR0915
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    to: Path | None = None,
    legend: bool = True,
    dpi: int = 200,
    large_legend: bool = False,
    outside_right_legend: bool = False,
    log: bool = False,
    xlines: list[float] | None = None,
    ylines: list[float] | None = None,
    highlight: list[str] | None = None,
    ymin: float = 0.0,
    sort_at: float | None = None,
) -> None:
    if to is None:
        to = Path("correlations.png")

    fig, ax = plt.subplots()
    assert isinstance(ax, plt.Axes)

    handles = []

    sorter = sort_at if sort_at else 0.0
    # Yup this looks awful, basically sorting by the mean at a particular index
    sorted_names = sorted(
        stats.keys(),
        key=lambda name: stats[name][0][int(sorter * len(stats[name][0]))],
        reverse=True,
    )
    markers = cycle(["o", "P", "X", "D", "v"])

    if highlight is None:
        n_colors = len(stats.items())
        colors = sns.color_palette("viridis", n_colors)
        alpha_line = 1
        alpha_std = 0.2
        for color, name, marker in zip(colors, sorted_names, markers):
            label = name
            mean, std = stats[name]
            xs = np.linspace(0, 1, len(mean))
            h = ax.plot(
                xs,
                mean,
                label=label,
                c=color,
                alpha=alpha_line,
                marker=marker,
                markersize=5,
            )
            ax.fill_between(xs, mean - std, mean + std, alpha=alpha_std, color=color)
            handles.append(h)
    else:
        # Sort the highlights by their index in the sorted_names above.
        highlight_names_sorted = sorted(
            highlight,
            key=lambda name: sorted_names.index(name),
        )
        colors = dict(
            zip(
                highlight_names_sorted,
                sns.color_palette("husl", len(highlight)),
            ),
        )

        # Plot the non highlighted stuff
        for name in stats:
            if name in colors:
                continue  # Skip anything plotted above
            mean, std = stats[name]
            xs = np.linspace(0, 1, len(mean))
            h = ax.plot(xs, mean, c="black", alpha=0.15)
            handles.append(h)

        # Plot the highlight stuff
        for (name, color), marker in zip(colors.items(), markers):
            label = name
            mean, std = stats[name]
            xs = np.linspace(0, 1, len(mean))
            h = ax.plot(
                xs,
                mean,
                label=label,
                c=color,
                alpha=1,
                marker=marker,
                markersize=5,
            )
            ax.fill_between(xs, mean - std, mean + std, alpha=0.1, color=color)
            handles.append(h)

    ax.set_xlim(auto=True)
    ax.set_ylim([ymin, 1])
    ax.set_ylabel("Spearman correlation", fontsize=18)
    ax.set_xlabel("Fidelity %", fontsize=18)

    xlines = xlines or []
    for _, x in enumerate(xlines, start=0):
        # y_offset = 0.05 if i % 2 == 0 else 0.10
        # x_text = f" {int(i * 100)}"
        alpha = 0.8 if sort_at is not None and np.isclose(x, sort_at) else 0.3
        ax.axvline(x, alpha=alpha, c="black", linestyle=":")
        # ax.text(x, y_offset, s=x_text, alpha=alpha, c="black", fontweight="bold")

    ylines = ylines or []
    for y in ylines:
        ax.axhline(y, xmin=0, xmax=1, c="black", linestyle="--")
        ax.text(
            0.005,
            y,
            s=f"{y}",
            c="black",
            horizontalalignment="left",
            verticalalignment="bottom",
            fontweight="bold",
        )

    if log:
        ax.set_xscale("log")

    ax.tick_params(axis="both", which="major", labelsize=15, labelcolor=(0, 0, 0, 0.69))
    ax.set_xticks([0.1, 1], labels=["10%", "100%"])
    ax.grid(True, which="major", ls="-", alpha=0.6)  # noqa: FBT003

    if legend:
        fsize = "x-large" if large_legend else "medium"
        if outside_right_legend:
            fig.legend(
                loc="center right",
                fontsize=fsize,
                bbox_to_anchor=(1.55, 0.5),
                frameon=True,
                ncol=1,
            )

        else:
            ax.legend(loc="lower right", fontsize=fsize)

    plt.tight_layout(pad=0, h_pad=0.5)
    plt.savefig(to, dpi=dpi, bbox_inches="tight")


def monte_carlo(
    benchmark: Benchmark,
    n_samples: int = 25,
    epsilon: float = 1e-3,
    iterations_max: int = 5000,
) -> RunningStats:
    """Compute the correlation curve use a mc method for convergence.

    Args:
        benchmark: The benchmark to compute the correlation curve for
        n_samples: The number of samples to take from the benchmark per iteration
        epsilon: The convergence threshold
        iterations_max: The maximum number of iterations to run

    Returns:
        RunningStats
    """
    stats = RunningStats()
    converged = False
    itrs = 0
    diff: float = np.inf
    while not converged and itrs < iterations_max:
        curve = correlation_curve(benchmark, n_samples=n_samples)
        stats.push(curve)

        if stats.n > 2:
            diff = float(np.linalg.norm(stats.new_m - stats.previous_m, ord=2))
            if diff <= epsilon:
                converged = True

        else:
            diff = np.inf
        itrs += 1

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--task_id", type=str)
    parser.add_argument("--datadir", type=str, required=False)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--iterations_max", type=int, default=MAX_ITERATIONS)

    parser.add_argument("--results_dir", type=str, default="correlation_results")

    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--plot_only", nargs="*", required=False)
    parser.add_argument("--plot_to", type=str, default="test.pdf")
    parser.add_argument("--plot_dpi", type=int, default=200)
    parser.add_argument("--no-legend", action="store_true", default=False)
    parser.add_argument("--large_legend", action="store_true", default=False)
    parser.add_argument("--outside_right_legend", action="store_true", default=False)
    parser.add_argument("--plot_log", action="store_true")
    parser.add_argument("--sort-at", type=float, default=0.11)
    parser.add_argument("--xlines", type=float, nargs="+", default=None)
    parser.add_argument("--ylines", type=float, nargs="+", default=None)
    parser.add_argument("--highlight", type=str, nargs="+", default=None)
    parser.add_argument("--ymin", type=float, default=0.0)

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        results_dir.mkdir()

    if args.plot:
        only = args.plot_only

        if only is None:
            names = [
                f.stem
                for f in results_dir.iterdir()
                if f.is_file() and f.suffix == ".json"
            ]
        else:
            names = only

        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in names:
            result_path = results_dir / f"{name}.json"
            with result_path.open("r") as f:
                result = json.load(f)
                results[name] = (np.array(result["mean"]), np.array(result["std"]))

        plot(
            results,
            to=args.plot_to,
            dpi=args.plot_dpi,
            legend=not args.no_legend,
            large_legend=args.large_legend,
            outside_right_legend=args.outside_right_legend,
            log=args.plot_log,
            sort_at=args.sort_at,
            xlines=args.xlines,
            ylines=args.ylines,
            highlight=args.highlight,
            ymin=args.ymin,
        )

    else:
        kwargs = {"name": args.benchmark, "seed": args.seed}

        if args.task_id:
            kwargs["task_id"] = args.task_id

        if args.datadir:
            datadir = Path(args.datadir)
            assert datadir.exists()
            kwargs["datadir"] = datadir

        b = mfpbench.get(**kwargs)
        stats = monte_carlo(
            benchmark=b,
            n_samples=args.n_samples,
            iterations_max=args.iterations_max,
            epsilon=args.epsilon,
        )

        results = {
            "mean": stats.mean().tolist(),  # type: ignore
            "std": stats.std().tolist(),
        }

        result_path = results_dir / f"{args.name}.json"
        with result_path.open("w") as f:
            json.dump(results, f)
