from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

import mfpbench
from mfpbench import Benchmark, MFHartmannBenchmark, YAHPOBenchmark

if TYPE_CHECKING:
    from mfpbench.result import Result


def benchmarks(
    *,
    seed: int,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    conditional_spaces: bool = False,  # Not supported due to `remove_hyperparamter`
) -> Iterator[Benchmark]:
    """Generate benchmarks.

    Args:
        seed: The seed to use for the benchmarks
        only: Only generate benchmarks that start with one of the strings in this list
        exclude: Exclude benchmarks that start with one of the strings in this list
        conditional_spaces: Whether to include benchmarks with conditional spaces

    Yields:
        The benchmarks
    """
    # A mapping from the indexable name to the argument name and cls
    benches: dict[str, tuple[str, type[Benchmark], str | None]] = {}

    for name, cls in mfpbench._mapping.items():
        if issubclass(cls, YAHPOBenchmark) and cls.yahpo_instances is not None:
            benches.update(
                {
                    f"{name}-{task_id}": (name, cls, task_id)
                    for task_id in cls.yahpo_instances
                },
            )
        else:
            benches[name] = (name, cls, None)

    for index_name, (benchmark_name, cls, task_id) in benches.items():
        if only is not None and not any(index_name.startswith(o) for o in only):
            continue

        if exclude is not None and any(index_name.startswith(e) for e in exclude):
            continue

        if cls.has_conditionals and not conditional_spaces:
            continue

        kwargs = {
            "name": benchmark_name,
            "seed": seed,
        }
        if task_id is not None:
            kwargs["task_id"] = task_id

        yield mfpbench.get(**kwargs)  # type: ignore


def generate_priors(  # noqa: C901
    *,
    seed: int,
    nsamples: int,
    to: Path,
    prior_spec: Iterable[tuple[str, int, float | None, float | None]],
    prefix: str | None = None,
    fidelity: int | float | None = None,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    use_hartmann_optimum: list[str] | None = None,
    clean: bool = False,
) -> None:
    """Generate priors for a benchmark."""
    if to.exists() and clean:
        for child in filter(lambda path: path.is_file(), to.iterdir()):
            child.unlink()

    to.mkdir(exist_ok=True)
    prior_spec = list(prior_spec)

    for bench in benchmarks(seed=seed, only=only, exclude=exclude):
        print(f" - Benchmark: {bench.name}")  # noqa: T201

        max_fidelity = bench.fidelity_range[1]

        # If a fidelity was specfied, then we need to make sure we can use it
        # as an int in a benchmark with an int fidelity, no accidental rounding.
        if fidelity is not None:
            if (
                isinstance(max_fidelity, int)
                and isinstance(fidelity, float)
                and fidelity.is_integer()
            ):
                fidelity = int(fidelity)

            if type(fidelity) != type(max_fidelity):
                raise ValueError(
                    f"Cannot use fidelity {fidelity} (type={type(fidelity)}) with"
                    f" benchmark {bench.name}",
                )
            at = fidelity
        else:
            at = max_fidelity

        results: list[Result] = []
        configs = bench.sample(n=nsamples)
        results = [bench.query(config, at=at) for config in configs]

        print(" - Finished results")  # noqa: T201
        results = sorted(results, key=lambda r: r.error)
        print(" - Finished sorting")  # noqa: T201

        # Take out the results as specified by the prior and store the perturbations
        # to make, if any.
        prior_configs = {
            name: (results[index].config, std, categorical_swap_chance)
            for name, index, std, categorical_swap_chance in prior_spec
        }

        # Inject hartmann optimum in if specified
        if use_hartmann_optimum is not None and isinstance(bench, MFHartmannBenchmark):
            for optimum_replace in use_hartmann_optimum:
                if optimum_replace not in prior_configs:
                    raise ValueError(f"Prior '{optimum_replace}' not found in priors.")

                opt = bench.optimum
                _, std, categorical_swap_chance = prior_configs[optimum_replace]

                prior_configs[optimum_replace] = (opt, std, categorical_swap_chance)

        print(" - Priors: ", prior_configs)  # noqa: T201

        # Perturb each of the configs as specified to make the offset priors
        space = bench.space
        priors = {
            name: config.perturb(
                space,
                seed=seed,
                std=std,
                categorical_swap_chance=categorical_swap_chance,
            )
            for name, (config, std, categorical_swap_chance) in prior_configs.items()
        }
        print(" - Perturbed priors: ", priors)  # noqa: T201

        name_components = []
        if prefix is not None:
            name_components.append(prefix)

        name_components.append(bench.name)

        basename = "-".join(name_components)

        path_priors = [
            (to / f"{basename}-{prior_name}.yaml", prior_config)
            for prior_name, prior_config in priors.items()
        ]
        for path, prior in path_priors:
            prior.save(path)
