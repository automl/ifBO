from __future__ import annotations

import json
import operator
import time
from dataclasses import dataclass, replace
from functools import reduce
from itertools import accumulate, chain, groupby, product, starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Iterable, Mapping, Sequence, overload

import yaml  # type: ignore
from joblib import Parallel, delayed
from more_itertools import all_equal, flatten, pairwise
from typing_extensions import Literal
from yaml import CLoader as Loader

if TYPE_CHECKING:
    import mfpbench
    import pandas as pd


def now() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def all_possibilities(
    experiment_group: str,
    base_path: Path,
    ignore_benchmarks: set[str] | None = None,
    ignore_algorithms: set[str] | None = None,
    ignore_seeds: set[int] | None = None,
) -> tuple[set[str], set[str], set[int]]:
    import re

    ignore_benchmarks = ignore_benchmarks or set()
    ignore_algorithms = ignore_algorithms or set()
    ignore_seeds = ignore_seeds or set()

    def is_ignored(s: str, ignore: set[str]) -> bool:
        return any(re.match(i, s) for i in ignore)

    RESULTS_DIR = base_path / "results" / experiment_group

    benchmarks = {p.name.split("=")[1] for p in RESULTS_DIR.glob("benchmark=*")}
    benchmarks = {b for b in benchmarks if not is_ignored(b, ignore_benchmarks)}

    algorithms = {p.name.split("=")[1] for p in RESULTS_DIR.glob("*/algorithm=*")}
    algorithms = {a for a in algorithms if not is_ignored(a, ignore_algorithms)}

    seeds = {int(p.name.split("=")[1]) for p in RESULTS_DIR.glob("*/*/seed=*")}
    seeds = seeds - ignore_seeds
    print(f"benchmarks={benchmarks}")
    print(f"algorithms={algorithms}")
    print(f"seeds={seeds}")
    return benchmarks, algorithms, seeds


def fetch_results(
    experiment_group: str,
    benchmarks: list[str],
    algorithms: list[str],
    base_path: Path,
    seeds: list[int] | None = None,
    n_workers: int = 1,
    parallel: bool = True,
    continuations: bool = True,
    cumulate_fidelities: bool = True,
    rescale: bool = True,
    rescale_xaxis: Literal["max_fidelity"] = "max_fidelity",
    incumbents_only: bool = True,
    incumbent_value: Literal["loss"] = "loss",
    xaxis: Literal[
        "cumulated_fidelity", "end_time_since_global_start"
    ] = "cumulated_fidelity",
    ignore_missing: bool = False,
) -> ExperimentResults:
    BENCHMARK_CONFIG_DIR = (
        base_path / "pfns_hpo" / "configs" / "benchmark"
    )
    if parallel:
        pool = Parallel(backend="multiprocessing", n_jobs=-1)
    else:
        pool = None

    RESULTS_DIR = base_path / "results" / experiment_group
    
    print(f"[{now()}]--- Loading results ---", flush=True)
    experiment_results = ExperimentResults.load(
        name=experiment_group,
        path=RESULTS_DIR,
        benchmarks=benchmarks,
        algorithms=algorithms,
        seeds=seeds,
        benchmark_config_dir=BENCHMARK_CONFIG_DIR,
        pool=pool,
        ignore_missing=ignore_missing,
    )
    
    if continuations:
        print(f"[{now()}]--- Calculating Continuations ---", flush=True)
        experiment_results = experiment_results.with_continuations(pool=pool)
    
    if cumulate_fidelities:
        print(f"[{now()}]--- Cumulating fidelities ---", flush=True)
        # fidelities: [1, 1, 3, 1, 9] -> [1, 2, 5, 6, 15]
        experiment_results = experiment_results.with_cumulative_fidelity(
            pool=pool, n_workers=n_workers
        )
    
    if incumbents_only:
        print(f"[{now()}]--- Getting incumbent traces ---", flush=True)
        # For now we only allow incumbent traces over "loss"
        experiment_results = experiment_results.incumbent_trace(
            xaxis=xaxis, yaxis=incumbent_value, pool=pool
        )
    
    if rescale and rescale_xaxis:
        print(f"[{now()}]--- Rescaling ---", flush=True)
        assert rescale_xaxis == "max_fidelity", "All we allow for now"
        experiment_results = experiment_results.rescale_xaxis(
            xaxis=xaxis, by=rescale_xaxis, pool=pool
        )
    
    return experiment_results


# NOTE: These need to be standalone functions for
# it to work with multiprocessing
def _with_continuations(a: AlgorithmResults) -> AlgorithmResults:
    return a.with_continuations()


def _with_cumulative_fidelity(
    a: AlgorithmResults, n_workers: int | None = None, algo_name: str | None = None
) -> AlgorithmResults:
    return a.with_cumulative_fidelity(n_workers=n_workers)


def _incumbent_trace(a: AlgorithmResults, xaxis: str, yaxis: str) -> AlgorithmResults:
    return a.incumbent_traces(xaxis=xaxis, yaxis=yaxis)


def _rescale_xaxis(a: AlgorithmResults, xaxis: str, c: float) -> AlgorithmResults:
    return a.rescale_xaxis(xaxis=xaxis, c=c)


def _in_range(
    a: AlgorithmResults, bounds: tuple[float, float], xaxis: str
) -> AlgorithmResults:
    return a.in_range(bounds=bounds, xaxis=xaxis)


def _algorithm_results(path: Path, seeds: list[int] | None) -> AlgorithmResults:
    return AlgorithmResults.load(path, seeds=seeds)


def _trace_results(
    path: Path, benchmark: str, algorithm: str, seed: int
) -> tuple[str, str, int, Trace]:
    print(benchmark, algorithm, seed)
    return benchmark, algorithm, seed, Trace.load(path)


@dataclass
class Result:
    id: int
    bracket: int | None
    loss: float
    cost: float
    val_score: float
    test_score: float
    fidelity: int
    start_time: float
    end_time: float
    max_fidelity_loss: float
    max_fidelity_cost: float
    min_valid_seen: float
    min_test_seen: float
    min_valid_ever: float
    min_test_ever: float
    cumulated_fidelity: float | None = None
    start_time_since_global_start: float | None = None
    end_time_since_global_start: float | None = None
    continued_from: Result | None = None
    process_id: int | None = None

    @classmethod
    def from_dir(cls, config_dir: Path) -> Result:
        result_yaml = config_dir / "result.yaml"
        with result_yaml.open("r") as f:
            result = yaml.load(f, Loader=Loader)

        config_name = config_dir.name.replace("config_", "")
        if "_" in config_name:
            id_, bracket = map(int, config_name.split("_"))
        else:
            id_ = int(config_name)
            bracket = None

        info = result["info_dict"]
        return cls(
            id=id_,
            bracket=bracket,
            loss=result["loss"],
            cost=result["cost"],
            val_score=info["val_score"],
            test_score=info["test_score"],
            fidelity=info["fidelity"],
            start_time=info["start_time"],
            end_time=info["end_time"],
            max_fidelity_loss=info["max_fidelity_loss"],
            max_fidelity_cost=info["max_fidelity_cost"],
            min_valid_seen=info["min_valid_seen"],
            min_test_seen=info["min_test_seen"],
            min_valid_ever=info["min_valid_ever"],
            min_test_ever=info["min_test_ever"],
            process_id=info.get("process_id"),
        )

    def continue_from(self, other: Result) -> Result:
        """Continue based on the results from a previous evaluation of the same config."""
        assert self.continued_from is None, f"{self} - {other}"
        changes = {
            "fidelity": self.fidelity - other.fidelity,
            "cost": self.cost - other.cost,
            "continued_from": other,
        }
        return self.mutate(**changes)

    def mutate(self, **kwargs: Any) -> Result:
        return replace(self, **kwargs)


@dataclass
class Trace(Sequence[Result]):
    results: list[Result]

    @classmethod
    def load(cls, path: Path) -> Trace:
        directories = list(path.iterdir())
        if path / "neps_root_directory" in directories:
            return cls.load_neps(path)
        elif path / "hpbandster_root_directory" in directories:
            return cls.load_hpbandster(path)
        else:
            raise ValueError(f"Neither neps/hpbandster_root_directory in {path}")

    @classmethod
    def load_hpbandster(cls, path: Path) -> Trace:
        result_dir = path / "hpbandster_root_directory"
        configs_file = result_dir / "configs.json"
        results_file = result_dir / "results.json"
        loaded_configs = {}
        with configs_file.open() as f:
            for line_ in f:
                line = json.loads(line_)
                if len(line) == 3:
                    id_, config, _ = line
                else:
                    id_, config = line

                hpbandster_config_id = tuple(id_)
                loaded_configs[hpbandster_config_id] = {"config": config}

        new_id_mappings = {
            hpbandster_config_id: i
            for i, hpbandster_config_id in enumerate(loaded_configs)
        }

        hpbandster_results = []
        with results_file.open("r") as f:
            for line_ in f:
                line = json.loads(line_)
                id_, budget, time_stamps, result, _ = line
                info = result["info"]
                id_ = tuple(id_)
                config = loaded_configs[id_]
                new_id = new_id_mappings[id_]

                hpbandster_results.append(
                    {
                        "config": {"id": new_id, "params": config, "bracket": None},
                        "start_time": time_stamps["started"],
                        "end_time": time_stamps["finished"],
                        "loss": result["loss"],
                        "cost": result["cost"],
                        "val_score": info["val_score"],
                        "test_score": info["test_score"],
                        "max_fidelity_loss": info["max_fidelity_loss"],
                        "max_fidelity_cost": info["max_fidelity_cost"],
                        "min_valid_seen": info["min_valid_seen"],
                        "min_test_seen": info["min_test_seen"],
                        "min_valid_ever": info["min_valid_ever"],
                        "min_test_ever": info["min_test_ever"],
                        "fidelity": int(budget),
                    }
                )

        unique_fidelities = set(r["fidelity"] for r in hpbandster_results)
        _fid_to_bracket = {f: i for i, f in enumerate(sorted(unique_fidelities))}

        for result in hpbandster_results:
            config = result["config"]
            id_ = config["id"]
            fidelity = result["fidelity"]
            bracket = _fid_to_bracket[fidelity]
            result["id"] = result["config"]["id"]
            result["bracket"] = bracket
            del result["config"]

        parsed_results = [Result(**r) for r in hpbandster_results]
        return cls(results=sorted(parsed_results, key=lambda r: r.end_time))

    @classmethod
    def load_neps(cls, path: Path) -> Trace:
        trace_results_dir = path / "neps_root_directory" / "results"

        assert trace_results_dir.exists(), f"Path {trace_results_dir} does not exist"
        config_dirs = [
            p for p in trace_results_dir.iterdir() if p.is_dir() and "config" in p.name
        ]
        results = list(map(Result.from_dir, config_dirs))

        if len(results) == 0:
            raise ValueError(f"Couldn't find results in {trace_results_dir}")

        global_start = min(result.start_time for result in results)
        results = [
            result.mutate(
                start_time_since_global_start=result.start_time - global_start,
                end_time_since_global_start=result.end_time - global_start,
            )
            for result in results
        ]

        results = sorted(results, key=lambda r: r.end_time)
        return cls(results=results)

    @overload
    def __getitem__(self, key: int) -> Result:
        ...

    @overload
    def __getitem__(self, key: slice) -> list[Result]:
        ...

    def __getitem__(self, key: int | slice) -> Result | list[Result]:
        return self.results[key]

    def __len__(self) -> int:
        return len(self.results)

    def indices(self, xaxis: str, *, sort: bool = True) -> list[float]:
        xs = [getattr(r, xaxis) for r in self.results]
        return xs if not sort else sorted(xs)

    def result_at(self, *, yaxis: str, budget: float) -> float:
        """Get the result at a given budget, ffill as needed."""
        index_not_over = -1
        for i, result in enumerate(self.results):
            assert result.cumulated_fidelity is not None
            if result.cumulated_fidelity > budget:
                break
            index_not_over = i
        result_closest_to_budget = self.results[index_not_over]

        return getattr(result_closest_to_budget, yaxis)

    @property
    def df(self) -> pd.DataFrame:
        import pandas as pd

        df = pd.DataFrame(
            data=[
                {
                    "loss": result.loss,
                    "cost": result.cost,
                    "val_score": result.val_score,
                    "test_score": result.test_score,
                    "fidelity": result.fidelity,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "max_fidelity_loss": result.max_fidelity_loss,
                    "max_fidelity_cost": result.max_fidelity_cost,
                    "min_valid_seen": result.min_valid_seen,
                    "min_test_seen": result.min_test_seen,
                    "min_valid_ever": result.min_valid_ever,
                    "min_test_ever": result.min_test_ever,
                    "cumulated_fidelity": result.cumulated_fidelity,
                    "config_id": str(result.id),
                    "continued_from": None
                    if result.continued_from is None
                    else f"{result.continued_from.id}_{result.continued_from.bracket}",
                    "bracket": str(result.bracket),
                    "process_id": result.process_id,
                }
                for result in self.results
            ]
        )
        df = df.set_index("end_time")
        assert df is not None
        df = df.sort_index(ascending=True)
        assert df is not None
        return df

    def with_continuations(self) -> Trace:
        """Add results for continuations of configs that were evaluated before."""
        # Group the results by the config id and then sort them by bracket
        # {
        #   0: [0_0, 0_1, 0_2]
        #   1: [1_0]
        #   2: [2_0, 2_1],
        # }
        def bracket(res: Result) -> int:
            return 0 if res.bracket is None else res.bracket

        # Needs to be sorted on the key before using groupby
        trace_results = sorted(self.results, key=lambda r: r.id)

        results = {
            config_id: sorted(results, key=bracket)
            for config_id, results in groupby(trace_results, key=lambda r: r.id)
        }

        continuations = []
        for config_results in results.values():
            # Put the lowest bracket entry into the continued results,
            # it can't have continued from anything
            continuations.append(config_results[0])

            if len(config_results) == 1:
                continue

            # We have more than one evaluation for this config (assumingly at a higher bracket)
            for lower_bracket, higher_bracket in pairwise(config_results):
                continued_result = higher_bracket.continue_from(lower_bracket)
                continuations.append(continued_result)

        assert len(trace_results) == len(continuations)

        sorted_continuations = sorted(continuations, key=lambda r: r.end_time)
        return replace(self, results=sorted_continuations)

    def with_cumulative_fidelity(
        self,
        n_workers: int | None = None,
        yaxis: str = "loss",
    ) -> Trace:
        """This only really makes sense for traces generated by single workers"""
        if n_workers is None or n_workers == 1:
            assert all_equal(r.process_id for r in self.results)
        else:
            assert all(r.process_id is not None for r in self.results)
            # unique_processes = {r.process_id for r in self.results}
            # assert len(unique_processes) == n_workers, f"{unique_processes}"

        if n_workers is None or n_workers == 1:
            results = sorted(self.results, key=lambda r: r.end_time)
            cumulated_fidelities = accumulate([r.fidelity for r in results])
            cumulated_results = [
                r.mutate(cumulated_fidelity=f)
                for r, f in zip(results, cumulated_fidelities)
            ]
        else:
            # NOTE: This is a fairly complex process
            results = sorted(
                self.results,
                key=lambda r: r.process_id if r.process_id is not None else 0,
            )
            # Group each processes list of results and make them each an individual trace
            # 0: Trace([...])
            # 1: [...]
            # 2: [...]
            # 3: [...]
            results_per_process = {
                pid: Trace(results=list(presults))
                for pid, presults in groupby(results, key=lambda r: r.process_id)
            }

            # Now for each processes trace, calculated the cumulated fidelities
            cumulated_results_per_process = {
                pid: trace.with_cumulative_fidelity()
                for pid, trace in results_per_process.items()
            }
            cumulated_results = []
            for trace in cumulated_results_per_process.values():
                cumulated_results.extend(trace.results)

            # NOTE: If removing this line and we expect someting
            # other than a loss, which needs to be minimized,
            # plese look for the `min` below and deal with it accordingly
            # i.e. if we used "score" instead, this `min` would need to be
            # a `max`
            assert yaxis == "loss", f"{yaxis} not supported right now"

            # Because we can have multiple results that share the same cumulated fidelity
            # now, we need to take the one with the best value according to the yaxis.

            # Sort by the cumulated_fidelity so groupby works
            cumulated_results = sorted(
                cumulated_results,
                key=lambda r: r.cumulated_fidelity,  # type: ignore
            )

            # For each list of results which share a cumulated_fidelity, get the
            # result with the minimum `yaxis` value (e.g. loss)
            cumulated_results = [
                min(results, key=lambda r: getattr(r, yaxis))
                for _, results in groupby(
                    cumulated_results, key=lambda r: r.cumulated_fidelity
                )
            ]

            # Finally resort tem according to cumulated fidelities so the timeline
            # is ordered
            cumulated_results = sorted(
                cumulated_results,
                key=lambda r: r.cumulated_fidelity,  # type: ignore
            )

        return replace(self, results=cumulated_results)

    def incumbent_trace(self, xaxis: str, yaxis: str) -> Trace:
        """Return a trace with only the incumbent results."""

        def _xaxis(r) -> float:
            return getattr(r, xaxis)

        if yaxis != "loss":
            raise NotImplementedError(f"yaxis={yaxis} not supported")

        results: list[Result] = sorted(self.results, key=_xaxis)

        incumbent = results[0]
        incumbents = [incumbent]
        for result in results[1:]:
            # If the new result is better than the incumbent, replace the incumbent
            if getattr(result, yaxis) < getattr(incumbent, yaxis):
                incumbent = result
                incumbents.append(incumbent)

        return replace(self, results=incumbents)

    def in_range(self, bounds: tuple[float, float], xaxis: str) -> Trace:
        low, high = bounds
        results = [
            result for resulct in self.results if low <= getattr(result, xaxis) <= high
        ]
        results = sorted(results, key=lambda r: getattr(r, xaxis))
        return replace(self, results=results)

    def rescale_xaxis(self, c: float, xaxis: str) -> Trace:
        results: list[Result] = []
        for result in self.results:
            copied = replace(result)
            value = getattr(result, xaxis)
            setattr(copied, xaxis, value * c)
            results.append(copied)

        results = sorted(results, key=lambda r: getattr(r, xaxis))
        return replace(self, results=results)

    def series(self, index: str, values: str, name: str | None = None) -> pd.Series:
        import pandas as pd

        indicies = [getattr(r, index) for r in self.results]
        vals = [getattr(r, values) for r in self.results]
        series = pd.Series(vals, index=indicies, name=name).sort_index()
        assert isinstance(series, pd.Series)
        return series


@dataclass
class Benchmark:
    name: str
    basename: str
    prior: str
    task_id: str | None
    optimum: float | None  # Only for mfh
    epsilon: float | None  # Only for some priors
    _config_path: Path
    _config: dict
    _benchmark: mfpbench.Benchmark | None  # Lazy loaded

    @classmethod
    def from_name(cls, name: str, config_dir: Path) -> Benchmark:
        expected_path = config_dir / f"{name}.yaml"
        if not expected_path.exists():
            raise ValueError(f"Expected benchmark path {expected_path} to exist.")

        with expected_path.open("r") as f:
            config = yaml.load(f, Loader=Loader)
        
        return cls(
            name=name,
            basename=config["api"]["name"],
            prior=config["api"]["prior"],
            epsilon=config["api"].get("epsilon"),
            task_id=config["api"].get("task_id"),
            optimum=config.get("optimum"),
            _config_path=expected_path,
            _config=config,
            _benchmark=None,
        )

    @property
    def benchmark(self) -> mfpbench.Benchmark:
        if self._benchmark is None:
            import mfpbench

            if self.task_id is not None:
                self._benchmark = mfpbench.get(self.basename, task_id=self.task_id)
            else:
                self._benchmark = mfpbench.get(self.basename)

        return self._benchmark

    @property
    def max_fidelity(self) -> int | float:
        return self.benchmark.end

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


@dataclass
class AlgorithmResults(Mapping[int, Trace]):
    traces: dict[int, Trace]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        seeds: list[int] | None = None,
    ) -> AlgorithmResults:
        """Load all traces for an algorithm."""
        if seeds is None:
            seeds = [
                int(p.name.split("=")[1])
                for p in path.iterdir()
                if p.is_dir() and "seed" in p.name
            ]

        paths = [path / f"seed={seed}" for seed in seeds]
        traces_ = list(map(Trace.load, paths))

        traces = {k: v for k, v in zip(seeds, traces_)}

        return cls(traces=traces)

    def select(self, seeds: list[int] | None = None) -> AlgorithmResults:
        if seeds is None:
            return replace(self)

        return replace(self, traces={s: self.traces[s] for s in seeds})

    def seeds(self) -> set[int]:
        return set(self.traces.keys())

    def indices(self, xaxis: str, *, sort: bool = True) -> list[float]:
        indices = [trace.indices(xaxis, sort=False) for trace in self.traces.values()]
        xs = list(set(flatten(indices)))
        return xs if not sort else sorted(xs)

    def with_continuations(self) -> AlgorithmResults:
        """Return a new AlgorithmResults with continuations."""
        traces = {seed: trace.with_continuations() for seed, trace in self.traces.items()}
        return replace(self, traces=traces)

    def with_cumulative_fidelity(self, n_workers: int | None = None) -> AlgorithmResults:
        traces = {}
        for seed, trace in self.traces.items():
            traces[seed] = trace.with_cumulative_fidelity(n_workers=n_workers)
        return replace(self, traces=traces)

    def incumbent_traces(
        self,
        xaxis: str,
        yaxis: str,
    ) -> AlgorithmResults:
        traces = {
            seed: trace.incumbent_trace(xaxis=xaxis, yaxis=yaxis)
            for seed, trace in self.traces.items()
        }
        return replace(self, traces=traces)

    def rescale_xaxis(self, xaxis: str, c: float) -> AlgorithmResults:
        traces = {
            seed: trace.rescale_xaxis(xaxis=xaxis, c=c)
            for seed, trace in self.traces.items()
        }
        return replace(self, traces=traces)

    def in_range(
        self,
        bounds: tuple[float, float],
        xaxis: str,
    ) -> AlgorithmResults:
        traces = {
            seed: trace.in_range(bounds=bounds, xaxis=xaxis)
            for seed, trace in self.traces.items()
        }
        return replace(self, traces=traces)

    def df(
        self,
        index: str,
        values: str,
        *,
        seeds: int | list[int] | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Return a dataframe with the traces."""
        import pandas as pd

        if seeds is None:
            traces = self.traces
        elif isinstance(seeds, int):
            traces = {seeds: self.traces[seeds]}
        else:
            traces = {seed: self.traces[seed] for seed in seeds}

        columns = [
            trace.series(index=index, values=values, name=f"seed-{seed}")
            for seed, trace in traces.items()
        ]
        df = pd.concat(columns, axis=1).sort_index(ascending=True)

        if len(traces) == 1:
            assert isinstance(df, pd.DataFrame)
            assert len(df.columns) == 1, df
            df = df[df.columns[0]]
            assert isinstance(df, pd.Series), f"{type(df)},\n{df}"
        else:
            assert isinstance(df, pd.DataFrame)

        return df

    def iter_results(self) -> Iterator[Result]:
        yield from chain.from_iterable(iter(trace) for trace in self.traces.values())

    def __getitem__(self, seed: int) -> Trace:
        return self.traces.__getitem__(seed)

    def __iter__(self) -> Iterator[int]:
        return self.traces.__iter__()

    def __len__(self) -> int:
        return self.traces.__len__()


@dataclass
class BenchmarkResults(Mapping[str, AlgorithmResults]):
    results: Mapping[str, AlgorithmResults]

    def select(
        self,
        *,
        algorithms: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> BenchmarkResults:
        if algorithms is None:
            selected = set(self.results.keys())
        else:
            selected = set(algorithms)

        results = {a: r.select(seeds) for a, r in self.results.items() if a in selected}
        return replace(self, results=results)

    def indices(self, xaxis: str, *, sort: bool = True) -> list[float]:
        indices = [algo.indices(xaxis, sort=False) for algo in self.results.values()]
        xs = list(set(flatten(indices)))
        return xs if not sort else sorted(xs)

    def seeds(self) -> set[int]:
        algo_seeds = [algo.seeds() for algo in self.results.values()]
        return set().union(*algo_seeds)

    def with_continuations(self, pool: Parallel | None = None) -> BenchmarkResults:
        keys = self.results.keys()
        results_: list[AlgorithmResults]
        if pool is not None:
            results_ = pool(delayed(_with_continuations)(v) for v in self.results.values())  # type: ignore
        else:
            results_ = list(map(_with_continuations, self.results.values()))

        results = {k: v for k, v in zip(keys, results_)}
        return replace(self, results=results)

    def with_cumulative_fidelity(
        self,
        n_workers: int | None = None,
        pool: Parallel | None = None,
    ) -> BenchmarkResults:
        keys = self.results.keys()
        results_: list[AlgorithmResults]
        args = [
            (algo_results, n_workers, algo_name)
            for algo_name, algo_results in self.results.items()
        ]
        if pool is not None:
            results_ = pool(delayed(_with_cumulative_fidelity)(*a) for a in args)  # type: ignore
        else:
            results_ = list(starmap(_with_cumulative_fidelity, args))
        results = {k: v for k, v in zip(keys, results_)}
        return replace(self, results=results)

    def incumbent_traces(
        self,
        xaxis: str,
        yaxis: str,
        *,
        pool: Parallel | None = None,
    ) -> BenchmarkResults:
        keys = self.results.keys()
        results_: list[AlgorithmResults]
        args = [(algo_results, xaxis, yaxis) for algo_results in self.results.values()]
        if pool is not None:
            results_ = pool(delayed(_incumbent_trace)(*a) for a in args)  # type: ignore
        else:
            results_ = list(starmap(_incumbent_trace, args))
        results = {k: v for k, v in zip(keys, results_)}
        return replace(self, results=results)

    def ranks(
        self,
        xaxis: str,
        yaxis: str,
        seed: int,
        indices: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        """Rank results for each algorithm on this benchmark for a certain seed."""
        import pandas as pd

        # NOTE: Everything here is in the context of a given seed for this
        # benchmark
        # {
        #   A1: [result_trace...]
        #   A2: [result_trace...]
        #   A3: [result_trace...]
        # }
        seed_results = {
            algo: results.select(seeds=[seed]) for algo, results in self.results.items()
        }

        # {
        #   A1: [incumbent_trace...]
        #   A2: [incumbent_trace...]
        #   A3: [incumbent_trace...]
        # }
        incumbent_results = {
            algo: results.incumbent_traces(xaxis=xaxis, yaxis="loss")
            for algo, results in seed_results.items()
        }

        # xaxis  |  A1,     A2,    A3
        #  1     |   .      .      .
        #  3     |   .      .      .
        #  6     |   .     nan     .
        #  9     |   .      .      .
        #  12    | nan      .      .
        series = [
            results.df(index=xaxis, values=yaxis).rename(algo)
            for algo, results in incumbent_results.items()
        ]
        df = pd.concat(series, axis=1)  # type: ignore

        # There may be a case where we want to merge this
        # rank table with another rank table and so we need
        # them to share the same set of indicies. Hence
        # we allow this as an option.
        if indices is not None:
            import numpy as np

            missing_indices = set(indices) - set(df.index)
            for index in missing_indices:
                df.loc[index] = np.nan

            df = df.sort_index(ascending=True)

        df = df.fillna(method="ffill", axis=0)
        df = df.rank(method="average", axis=1)
        return df

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        pool: Parallel | None = None,
        algorithms: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> BenchmarkResults:
        if algorithms is None:
            algorithms = [
                p.name.split("=")[1]
                for p in path.iterdir()
                if p.is_dir() and "algo" in p.name
            ]

        args = [(path / f"algorithm={algo}", seeds) for algo in algorithms]

        if pool is not None:
            results = pool(delayed(_algorithm_results)(*a) for a in args)
        else:
            results = list(starmap(_algorithm_results, args))

        return cls(results=dict(zip(algorithms, results)))  # type: ignore

    def rescale_xaxis(
        self,
        xaxis: str,
        c: float,
        *,
        pool: Parallel | None = None,
    ) -> BenchmarkResults:
        keys = self.results.keys()
        args = [(algo_results, xaxis, c) for algo_results in self.results.values()]
        if pool is not None:
            results_ = pool(delayed(_rescale_xaxis)(*a) for a in args)
        else:
            results_ = list(starmap(_rescale_xaxis, args))
        results = {k: v for k, v in zip(keys, results_)}  # type: ignore
        return replace(self, results=results)

    def in_range(
        self,
        bounds: tuple[float, float],
        xaxis: str,
        *,
        pool: Parallel | None = None,
    ) -> BenchmarkResults:
        keys = self.results.keys()
        args = [(algo_results, bounds, xaxis) for algo_results in self.results.values()]
        if pool is not None:
            results_ = pool(delayed(_in_range)(*a) for a in args)
        else:
            results_ = list(starmap(_in_range, args))
        results = {k: v for k, v in zip(keys, results_)}  # type: ignore
        return replace(self, results=results)

    def iter_results(self) -> Iterator[Result]:
        yield from chain.from_iterable(
            algo_results.iter_results() for algo_results in self.results.values()
        )

    def __getitem__(self, algo: str) -> AlgorithmResults:
        return self.results.__getitem__(algo)

    def __iter__(self) -> Iterator[str]:
        return self.results.__iter__()

    def __len__(self) -> int:
        return self.results.__len__()


@dataclass
class ExperimentResults(Mapping[str, BenchmarkResults]):
    name: str
    algorithms: list[str]
    benchmarks: list[str]
    benchmark_configs: dict[str, Benchmark]
    results: dict[str, BenchmarkResults]

    @classmethod
    def load(
        cls,
        name: str,
        path: Path,
        *,
        benchmarks: list[str],
        algorithms: list[str],
        seeds: list[int] | None = None,
        benchmark_config_dir: Path,
        pool: Parallel | None = None,
        ignore_missing: bool = False,
    ) -> ExperimentResults:
        if seeds is None:
            seeds = sorted(int(p.name.split("=")[1]) for p in path.glob("*/*/seed=*"))

        def _path(benchmark_: str, algorithm_: str, seed_: int) -> Path:
            return (
                path
                / f"benchmark={benchmark_}"
                / f"algorithm={algorithm_}"
                / f"seed={seed_}"
            )

        if pool is None:
            pool = Parallel(n_jobs=1)

        # path, benchmark, algorithm, seed
        items = (
            (_path(b, a, s), b, a, s)
            for b, a, s in product(benchmarks, algorithms, seeds)
        )
        if ignore_missing:
            items = ((p, b, a, s) for p, b, a, s in items if p.exists())
    
        # paralllel collection of results from each path
        ## TIME CONSUMING BLOCK
        parallel_results: list[tuple[str, str, int, Trace]] = pool(
            delayed(_trace_results)(p, b, a, s) for p, b, a, s in items
        )  # type: ignore
        # end of data collection as a list of result dicts
        results = {}
        for benchmark, algorithm, seed, trace in parallel_results:
            # creates a nested dict containing {benchmark: {algorithm: {seed: trace}}}
            results.setdefault(benchmark, {}).setdefault(algorithm, {})[seed] = trace
        return cls(
            name=name,
            algorithms=algorithms,
            benchmarks=benchmarks,
            results={
                benchmark: BenchmarkResults(
                    {
                        algo: AlgorithmResults(
                            {seed: trace for seed, trace in algo_results.items()}
                        )
                        for algo, algo_results in benchmark_results.items()
                    }
                )
                for benchmark, benchmark_results in results.items()
            },
            benchmark_configs={
                benchmark: Benchmark.from_name(benchmark, benchmark_config_dir)
                for benchmark in benchmarks
            },
        )

    def indices(self, xaxis: str, *, sort: bool = True) -> list[float]:
        indices = [bench.indices(xaxis, sort=False) for bench in self.results.values()]
        xs = list(set(flatten(indices)))
        return xs if not sort else sorted(xs)

    def with_continuations(self, pool: Parallel | None = None) -> ExperimentResults:
        results = {k: v.with_continuations(pool) for k, v in self.results.items()}
        return replace(self, results=results)

    def with_cumulative_fidelity(
        self,
        n_workers: int | None = None,
        pool: Parallel | None = None,
    ) -> ExperimentResults:
        results = {}
        for k, v in self.results.items():
            results[k] = v.with_cumulative_fidelity(n_workers=n_workers, pool=pool)

        return replace(self, results=results)

    def incumbent_trace(
        self,
        xaxis: str,
        yaxis: str,
        *,
        pool: Parallel | None = None,
    ) -> ExperimentResults:
        results = {
            k: v.incumbent_traces(xaxis=xaxis, yaxis=yaxis, pool=pool)
            for k, v in self.results.items()
        }
        return replace(self, results=results)

    def rescale_xaxis(
        self,
        xaxis: str,
        by: Literal["max_fidelity"],
        *,
        pool: Parallel | None = None,
    ) -> ExperimentResults:
        if by != "max_fidelity":
            raise NotImplementedError(f"by={by}")

        max_fidelities = {
            name: benchmark.max_fidelity
            for name, benchmark in self.benchmark_configs.items()
        }

        results = {
            name: benchmark_results.rescale_xaxis(
                xaxis=xaxis, c=(1 / max_fidelities[name]), pool=pool
            )
            for name, benchmark_results in self.results.items()
        }
        return replace(self, results=results)

    def in_range(
        self,
        bounds: tuple[float, float],
        xaxis: str,
        *,
        pool: Parallel | None = None,
    ) -> ExperimentResults:
        results = {
            k: v.in_range(bounds=bounds, xaxis=xaxis, pool=pool)
            for k, v in self.results.items()
        }
        return replace(self, results=results)

    def __getitem__(self, benchmark: str) -> BenchmarkResults:
        return self.results.__getitem__(benchmark)

    def __iter__(self) -> Iterator[str]:
        return self.results.__iter__()

    def __len__(self) -> int:
        return self.results.__len__()

    def iter_results(self) -> Iterator[Result]:
        yield from chain.from_iterable(
            benchmark_results.iter_results()
            for benchmark_results in self.results.values()
        )

    def select(
        self,
        *,
        benchmarks: list[str] | None = None,
        algorithms: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> ExperimentResults:
        if benchmarks is None:
            benchmarks = list(set(self.benchmarks))
        else:
            benchmark_set = set(benchmarks)
            benchmarks = sorted(
                benchmark_set, key=lambda b, benchmarks=benchmarks: benchmarks.index(b)
            )

        selected_results = {
            name: benchmark.select(algorithms=algorithms, seeds=seeds)
            for name, benchmark in self.results.items()
            if name in benchmarks
        }
        benchmark_configs = {
            name: config
            for name, config in self.benchmark_configs.items()
            if name in benchmarks
        }

        if algorithms is None:
            algorithms = self.algorithms

        return replace(
            self,
            name=self.name,
            benchmarks=benchmarks,
            algorithms=algorithms,
            results=selected_results,
            benchmark_configs=benchmark_configs,
        )

    def seeds(self) -> set[int]:
        bench_seeds = [bench.seeds() for bench in self.results.values()]
        return set().union(*bench_seeds)

    def ranks(self, *, xaxis: str, yaxis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        import pandas as pd

        indices = self.indices(xaxis=xaxis, sort=False)
        seeds = self.seeds()
        benchmarks = self.benchmarks

        # Get the ranks for each seed, benchmark possible
        ranks = {
            seed: {
                benchmark: self.results[benchmark].ranks(
                    xaxis, yaxis, seed=seed, indices=indices
                )
                for benchmark in benchmarks
            }
            for seed in seeds
        }

        # Utility to calculate the mean and the sem
        # Done by stacking all dataframes such that their index (i.e. cumulated fidelity)
        # are duplicated.
        #
        #       Dataframe 1
        # index | A1  A2  A3
        #  1
        #  2
        #  3
        #  1
        #  2
        #  3
        #
        # Then we move the index to be a column that we can then groupby
        # which collects all dataframes with the same index together
        #
        # Groupby over multiple Dataframes [1, 2, 3]
        # ---  index   dfs...
        #  0       1   [A1 A2 A3], [A1 A2 A3]
        #  1       2   [A1 A2 A3], [A1 A2 A3]
        #  2       3   ...
        #
        # We can then just call mean and sem on them
        def _mean(_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(_dfs).reset_index().groupby("index").mean()

        def _sem(_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(_dfs).reset_index().groupby("index").sem()

        # Get mean across all benchmarks, for each seed
        ranks_per_seed_averaged_over_benchmarks = {
            seed: _mean(ranks[seed].values()) for seed in seeds
        }

        # Average over all seeds
        mean_ranks = _mean(ranks_per_seed_averaged_over_benchmarks.values())
        sem_ranks = _sem(ranks_per_seed_averaged_over_benchmarks.values())

        return mean_ranks, sem_ranks

    def table_results(
        self,
        *,
        xs: list[int],
        yaxis: str,
        sort_order: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        import pandas as pd

        budgets = xs

        benchmarks = self.benchmarks
        algorithms = self.algorithms
        seeds = self.seeds()

        # Sorry for the complex pandas magic, this made sense at
        # the time of writing
        dataframes_by_seed = {
            seed: pd.DataFrame(
                {
                    (budget, algorithm): [
                        self.results[benchmark]
                        .results[algorithm]
                        .traces[seed]
                        .result_at(budget=budget, yaxis=yaxis)
                        for benchmark in benchmarks
                    ]
                    for budget, algorithm in product(budgets, algorithms)
                },
                index=benchmarks
            )
            for seed in seeds
        }
        for df in dataframes_by_seed.values():
            df.columns = pd.MultiIndex.from_tuples(df.columns)

        results_grouped_by_benchmarks = pd.concat(dataframes_by_seed.values()).groupby(level=0)
        means = results_grouped_by_benchmarks.agg("mean")
        stds = results_grouped_by_benchmarks.agg("std")

        def prior_order(index: pd.Index) -> pd.Index:
            prior_name = lambda bench_name: bench_name.rsplit("-", 1)[-1]
            return index.map(lambda bench_name: sort_order.index(prior_name(bench_name)))

        means.sort_index(inplace=True, key=prior_order)
        stds.sort_index(inplace=True, key=prior_order)
        return means, stds  # type: ignore
