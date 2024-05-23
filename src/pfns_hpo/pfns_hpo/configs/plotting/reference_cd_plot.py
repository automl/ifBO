import errno
import math
import operator
import os
import time
import warnings
from multiprocessing import Manager

import matplotlib.pyplot as plt
import networkx
import numpy as np
import pandas as pd
from attrdict import AttrDict
from joblib import Parallel, delayed, parallel_backend
from path import Path
from scipy.stats import friedmanchisquare, wilcoxon

from .configs.plotting.read_results import get_seed_info, SINGLE_FIDELITY_ALGORITHMS
from .configs.plotting.utils import (
    ALGORITHMS,
    get_max_fidelity,
    get_parser,
    interpolate_time,
    save_fig,
)
from .plot import map_axs

benchmark_configs_path = os.path.join(os.path.dirname(__file__), "configs/benchmark/")

# Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
#         Germain Forestier <germain.forestier@uha.fr>
#         Jonathan Weber <jonathan.weber@uha.fr>
#         Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
#         Pierre-Alain Muller <pierre-alain.muller@uha.fr>
# License: GPL3

# Adopted from TabPFN codebase https://github.com/automl/TabPFN

ALGORITHM_COLUMN_NAME = "algorithm"
TASK_COLUMN_NAME = "benchmark"
METRIC_COLUMN_NAME = "loss"


### CD Graph utils

# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(
    ax,
    avranks,
    names,
    p_values,
    lowv=None,
    highv=None,
    width=6,
    textspace=1,
    reverse=False,
    labels=False,
):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not lr:
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    # def print_figure(fig, *args, **kwargs):
    #     canvas = FigureCanvasAgg(fig)
    #     canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    # lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            _a = rank - lowv
        else:
            _a = highv - rank

        if highv - lowv == 0:
            warnings.warn("divisor was 0, (highv - lowv)")
            return textspace

        if _a == 0:
            warnings.warn("divisor was 0, (_a)")
            return textspace

        return textspace + scalewidth / (highv - lowv) * _a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    # fig = plt.figure(figsize=(width, height))
    # fig.set_facecolor('white')
    # ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1.0 / height  # height factor
    wf = 1.0 / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color="k", **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=2)

    for a in range(lowv, highv + 1):
        text(
            rankpos(a), cline - tick / 2 - 0.05, str(a), ha="center", va="bottom", size=16
        )

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace - 0.1, chei),
            ],
            linewidth=linewidth,
        )
        if labels:
            text(
                textspace + 0.3,
                chei - 0.075,
                format(ssums[i], ".2f"),
                ha="right",
                va="center",
                size=10,
            )
        text(
            textspace - 0.2,
            chei,
            filter_names(nnames[i]),
            ha="right",
            va="center",
            size=16,
        )

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace + scalewidth + 0.1, chei),
            ],
            linewidth=linewidth,
        )
        if labels:
            text(
                textspace + scalewidth - 0.3,
                chei - 0.075,
                format(ssums[i], ".2f"),
                ha="left",
                va="center",
                size=10,
            )
        text(
            textspace + scalewidth + 0.2,
            chei,
            filter_names(nnames[i]),
            ha="left",
            va="center",
            size=16,
        )

    # no-significance lines
    # def draw_lines(lines, side=0.05, height=0.1):
    #     start = cline + 0.2
    #
    #     for l, r in lines:
    #         line(
    #             [(rankpos(ssums[l]) - side, start), (rankpos(ssums[r]) + side, start)],
    #             linewidth=linewidth_sign,
    #         )
    #         start += height
    #         print("drawing: ", l, r)

    # draw_lines(lines)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    # print(nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        # print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half is False:
            start = cline + 0.25
            achieved_half = True
        line(
            [
                (rankpos(ssums[min_idx]) - side, start),
                (rankpos(ssums[max_idx]) + side, start),
            ],
            linewidth=linewidth_sign,
        )
        start += height
    return ax


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] is False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def get_classifier_comparison_df(df_perf, performance_metric_column_name, classifiers):
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    better_classifiers = []
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(
            df_perf.loc[df_perf[ALGORITHM_COLUMN_NAME] == classifier_1][
                performance_metric_column_name
            ],
            dtype=np.float64,
        )
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(
                df_perf.loc[df_perf[ALGORITHM_COLUMN_NAME] == classifier_2][
                    performance_metric_column_name
                ],
                dtype=np.float64,
            )
            diff_perf = perf_1 - perf_2
            wins = sum(diff_perf > 0)
            tie = sum(diff_perf == 0)
            loss = sum(diff_perf < 0)
            if wins > loss:
                winner = classifier_1
            else:
                winner = classifier_2
            # calculate the p_value
            p_value = np.around(
                wilcoxon(perf_1, perf_2, zero_method="pratt")[1], decimals=4
            )
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
            better_classifiers.append(
                {
                    "classifier_1": classifier_1,
                    "classifier_2": classifier_2,
                    "p_value": p_value,
                    "winner": winner,
                    "wins": wins,
                    "tie": tie,
                    "loss": loss,
                }
            )
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))
    better_classifiers.sort(key=operator.itemgetter("p_value"))
    better_classifiers = pd.DataFrame(better_classifiers)

    return better_classifiers, p_values


def get_classifiers(df_perf):
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame(
        {"count": df_perf.groupby([ALGORITHM_COLUMN_NAME]).size()}
    ).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts["count"].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(
        df_counts.loc[df_counts["count"] == max_nb_datasets][ALGORITHM_COLUMN_NAME]
    )
    return df_counts, max_nb_datasets, classifiers


# calculate wilcoxon holm p-value
def wilcoxon_holm(df_perf, performance_metric_column_name="roc_auc", alpha=0.05):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    _, max_nb_datasets, classifiers = get_classifiers(df_perf)
    # get the number of classifiers
    m = len(classifiers)
    # test the null hypothesis using friedman before doing a post-hoc analysis
    try:
        friedman_p_value = friedmanchisquare(
            *(
                np.array(
                    df_perf.loc[df_perf[ALGORITHM_COLUMN_NAME] == c][
                        performance_metric_column_name
                    ]
                )
                for c in classifiers
            )
        )[1]
        if friedman_p_value >= alpha:
            # then the null hypothesis over the entire classifiers cannot be rejected
            warnings.warn(
                f"The null hypothesis over the entire classifiers: {classifiers} cannot be rejected"
            )
    except:  # pylint: disable=bare-except
        pass

    _, p_values = get_classifier_comparison_df(
        df_perf=df_perf,
        performance_metric_column_name=performance_metric_column_name,
        classifiers=classifiers,
    )
    # get the number of hypothesis
    k = len(p_values)

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (
                p_values[i][0],
                p_values[i][1],
                p_values[i][2],
                p_values[i][2] <= new_alpha,
            )
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[
        df_perf[ALGORITHM_COLUMN_NAME].isin(classifiers)
    ].sort_values([ALGORITHM_COLUMN_NAME, TASK_COLUMN_NAME])
    # get the rank data
    rank_data = np.array(sorted_df_perf[performance_metric_column_name]).reshape(
        m, max_nb_datasets
    )

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(
        data=rank_data,
        index=np.sort(classifiers),
        columns=np.unique(sorted_df_perf[TASK_COLUMN_NAME]),
    )

    # # number of wins
    # dfff = df_ranks.rank(ascending=False)
    # # print(dfff)  # [dfff == 1.0].sum(axis=1))
    # # dfff.T.to_csv(os.path.join(out_dir, "rank_per_task.csv"))
    # average the ranks
    average_ranks = (
        df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    )
    # average_ranks.T.to_csv(os.path.join(out_dir, "average_ranks_per_task.csv"))
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets


def _process_seed(
    _path,
    seed,
    algorithm,
    key_to_extract,
    cost_as_runtime,
    results,
    n_workers,
    parallel_sleep_decrement,  # Needed for parallel setups
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
        parallel_sleep_decrement=parallel_sleep_decrement,
    )
    incumbent = np.minimum.accumulate(losses)
    cost = [i[key_to_extract] for i in infos]
    results["incumbents"].append(incumbent)
    results["costs"].append(cost)
    results["max_costs"].append(max_cost)


def plot(args):
    assert args.budget is not None, "Please input the budget for which CD is calculated"

    starttime = time.time()

    BASE_PATH = (
        Path(__file__).parent / "../.."
        if args.base_path is None
        else Path(args.base_path)
    )

    KEY_TO_EXTRACT = "cost" if args.cost_as_runtime else "fidelity"

    ncols = len(args.budget)
    nrows = 1
    figsize = (4 * ncols, 3 * nrows)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
    )

    base_path = BASE_PATH / "results" / args.experiment_group
    output_dir = BASE_PATH / "plots" / args.experiment_group

    for budget_idx, budget in enumerate(args.budget):

        ax = map_axs(axs, budget_idx, len(args.budget), ncols)

        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}]"
            f" Processing {len(args.benchmarks)} benchmarks "
            f"and {len(args.algorithms)} algorithms..."
        )

        df_perf = pd.DataFrame(
            columns=[TASK_COLUMN_NAME, METRIC_COLUMN_NAME, ALGORITHM_COLUMN_NAME]
        )
        for benchmark_idx, benchmark in enumerate(args.benchmarks):

            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] "
                f"[{benchmark_idx}] Processing {benchmark} benchmark..."
            )
            benchmark_starttime = time.time()

            _base_path = os.path.join(base_path, f"benchmark={benchmark}")
            if not os.path.isdir(_base_path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), _base_path
                )
            for algorithm in args.algorithms:
                _path = os.path.join(_base_path, f"algorithm={algorithm}")
                if not os.path.isdir(_path):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), _path
                    )

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
                            args.n_workers,
                            args.parallel_sleep_decrement,
                        )
                        for seed in seeds
                    ]

                incumbents = np.array(results["incumbents"][:])
                costs = np.array(results["costs"][:])
                max_cost = None if args.cost_as_runtime else max(results["max_costs"][:])

                if args.n_workers > 1 and max_cost is None:
                    max_cost = get_max_fidelity(benchmark_name=benchmark)

                df = interpolate_time(
                    incumbents,
                    costs,
                    scale_x=max_cost,
                    parallel_evaluations=(args.n_workers > 1),
                    # Single fidelity runners may incur some slight machine overhead but
                    # in practice and theory, this would be non-existent or neglible, they
                    # will always have a unit cost of an epoch
                    rounded_integer_costs_for_x_range=(algorithm in SINGLE_FIDELITY_ALGORITHMS)
                )

                if budget is not None:
                    pre_index = df.index
                    if algorithm in SINGLE_FIDELITY_ALGORITHMS:
                        df.index = df.index.astype(int)

                    df = df.query(f"index <= {budget}")
                    if len(df) == 0:
                        raise ValueError(
                            f"No values in dataframe index were below the budget: {budget}."
                            f"\nalgorithm: {algorithm} | benchmark: {benchmark}"
                            f"\nThe costs recorded in dataframe index are {pre_index}"
                        )

                # Take mean across seeds
                final_mean = df.mean(axis=1).values[-1]

                df_perf.loc[len(df_perf)] = [benchmark, final_mean, ALGORITHMS[algorithm]]

                print(
                    f"Time to process algorithm data: {time.time() - algorithm_starttime}"
                )
            print(f"Time to process benchmark data: {time.time() - benchmark_starttime}")

        p_values, average_ranks, _ = wilcoxon_holm(
            df_perf=df_perf, performance_metric_column_name=METRIC_COLUMN_NAME
        )

        ax = graph_ranks(
            ax,
            [round(float(value), 2) for value in list(average_ranks.values)],
            average_ranks.keys(),
            p_values,
            reverse=True,
            width=9,
            textspace=1.5,
            labels=False,
        )

        ax.set_title(f"{args.plot_id}@{int(budget)}", fontsize=20)

    fig.tight_layout(pad=0, h_pad=0.5)

    filename = args.filename
    if filename is None:
        filename = f"{args.experiment_group}_{args.plot_id}_cd_diagram"
    save_fig(
        fig,
        filename=filename,
        output_dir=output_dir,
        extension=args.ext,
        dpi=args.dpi,
    )
    # output_dir = Path(output_dir)
    # output_dir.makedirs_p()

    print(f"Plotting took {time.time() - starttime}")


if __name__ == "__main__":
    parser = get_parser()
    args = AttrDict(parser.parse_args().__dict__)
    plot(args)  # pylint: disable=no-value-for-parameter
