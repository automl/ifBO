"""A cli entry point."""
from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from typing_extensions import override

import mfpbench
import mfpbench.priors
import mfpbench.setup_benchmark


@dataclass
class CommandHandler(ABC):
    """A handler for a command."""

    name: ClassVar[str]
    help: ClassVar[str]

    @classmethod
    @abstractmethod
    def do(cls, args: argparse.Namespace) -> None:
        """Handle the command."""
        ...

    @classmethod
    @abstractmethod
    def fill_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Parser for the command."""
        ...

    @classmethod
    def get(cls) -> list[type[CommandHandler]]:
        """Get all the handlers."""
        return cls.__subclasses__()


class BenchmarkInstallHandler(CommandHandler):
    name = "install"
    help = "Install requirements for use with mfpbench."

    @classmethod
    def do(cls, args: argparse.Namespace) -> None:
        """Download the data."""
        if args.list:
            for source in mfpbench.setup_benchmark.BenchmarkSetup.sources():
                print(source.name)
            return

        if args.view:
            mfpbench.setup_benchmark.print_requirements(args.benchmark)
            return

        if args.benchmark is None:
            print("Please specify a --benchmark to install")
            return

        mfpbench.setup_benchmark.setup(
            benchmark=args.benchmark,
            download=False,
            install=True if args.requirements is None else args.requirements,
        )

    @override
    @classmethod
    def fill_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--list",
            action="store_true",
            help="Print out the available benchmarks data sources",
        )
        parser.add_argument(
            "--view",
            action="store_true",
            help="View the default requirements for a benchmark",
        )
        parser.add_argument(
            "--benchmark",
            choices=[
                source.name
                for source in mfpbench.setup_benchmark.BenchmarkSetup.sources()
            ],
            help="The benchmark to setup.",
        )
        parser.add_argument(
            "--requirements",
            type=str,
            help=(
                "The requirements into the currently active environment."
                " If not specified, will use the default requirements."
                " If string, the string will be used as the path to the requirements"
                " file."
            ),
            default=None,
        )
        return parser


class BenchmarkDownloadHandler(CommandHandler):
    name = "download"
    help = "Download datasets for use with mfpbench."

    @classmethod
    def do(cls, args: argparse.Namespace) -> None:
        """Download the data."""
        if args.status:
            mfpbench.setup_benchmark.print_download_status(
                sources=args.benchmark,
                datadir=args.data_dir,
            )
            return

        if args.list:
            for source in mfpbench.setup_benchmark.BenchmarkSetup.sources():
                print(source.name)
            return

        if args.benchmark is None:
            print("Please specify a --benchmark to download")
            return

        mfpbench.setup_benchmark.setup(
            benchmark=args.benchmark,
            datadir=args.data_dir,
            download=True,
            install=False,
            force=args.force,
            workers=args.workers,
        )

    @override
    @classmethod
    def fill_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force download and remove existing data",
        )
        parser.add_argument(
            "--status",
            action="store_true",
            help="Print out the status of benchmarks",
        )
        parser.add_argument(
            "--list",
            action="store_true",
            help="Print out the available benchmarks data sources",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help=(
                "The number of workers to use for downloading"
                " if the downlaoder supports it"
            ),
        )
        parser.add_argument(
            "--benchmark",
            choices=[
                source.name
                for source in mfpbench.setup_benchmark.BenchmarkSetup.sources()
            ],
            type=str,
            default=None,
            help="The benchmark to download.",
        )
        parser.add_argument(
            "--data-dir",
            default=mfpbench.setup_benchmark.DATAROOT,
            type=Path,
            help=(
                "Where to save the data,"
                f" defaults to './{mfpbench.setup_benchmark.DATAROOT}'"
            ),
        )
        return parser


class GeneratePriorsHandler(CommandHandler):
    name = "priors"
    help = "Generate priors for use with mfpbench."

    @override
    @classmethod
    def do(cls, args: argparse.Namespace) -> None:
        """Generate priors."""
        mfpbench.priors.generate_priors(
            seed=args.seed,
            nsamples=args.nsamples,
            prefix=args.prefix,
            to=args.to,
            fidelity=args.fidelity,
            only=args.only,
            exclude=args.exclude,
            prior_spec=args.priors,
            clean=args.clean,
            use_hartmann_optimum=args.use_hartmann_optimum,
        )

    @classmethod
    def parse_prior(cls, s: str) -> tuple[str, int, float | None, float | None]:
        """Parse a prior string."""
        name, index, noise, categorical_swap_chance = s.split(":")
        try:
            _index = int(index)
        except ValueError as e:
            raise ValueError(f"Invalid index {index}") from e

        if noise in ("None", "0", "0.0", "0.00"):
            _noise = None
        else:
            try:
                _noise = float(noise)
                if not (0 <= _noise <= 1):
                    raise ValueError(f"noise must be in [0, 1] in ({name}:{_noise})")
            except ValueError as e:
                raise TypeError(
                    f"Can't convert {noise} to float in ({name}:{noise})",
                ) from e

        if categorical_swap_chance in ("None", "0", "0.0", "0.00"):
            _categorical_swap_chance = None
        else:
            try:
                _categorical_swap_chance = float(categorical_swap_chance)
                if not (0 <= _categorical_swap_chance <= 1):
                    raise ValueError(
                        f"categorical_swap_chance must be in [0, 1] in ({s})",
                    )
            except ValueError as e:
                raise TypeError(
                    f"Can't convert categorical_swap_chance ({categorical_swap_chance})"
                    f" to float in ({s})",
                ) from e

        return name, _index, _noise, _categorical_swap_chance

    @override
    @classmethod
    def fill_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--seed", type=int, default=133_077, help="The seed to use")
        parser.add_argument(
            "--nsamples",
            type=int,
            default=1_000_000,
            help="The number of samples to generate",
        )
        parser.add_argument(
            "--prefix",
            type=str,
            help="The prefix to use for the generated prior file",
        )
        parser.add_argument(
            "--to",
            type=Path,
            default=Path("priors"),
            help="Where to save the priors",
        )
        parser.add_argument(
            "--fidelity",
            type=float,
            required=False,
            help="The fidelity to evaluated at, defaults to max fidelity",
        )
        parser.add_argument(
            "--priors",
            type=cls.parse_prior,
            nargs="+",
            help=(
                "The <priorname>:<index>:<std>:<categorical_swap_chance>"
                " of the priors to generate. You can use python's negative"
                " indexing to index from the end. If a value for std or"
                " categorical_swap_chance is 0 or None, then it will not"
                " be used. However it must be specified."
            ),
            default=[
                ("good", 0, 0.01, None),
                ("medium", 0, 0.125, None),
                ("bad", -1, None, None),
            ],
        )
        parser.add_argument(
            "--only",
            type=str,
            nargs="*",
            help="Only generate priors for these benchmarks",
        )
        parser.add_argument(
            "--exclude",
            type=str,
            nargs="*",
            help="Exclude benchmarks",
        )
        parser.add_argument(
            "--clean",
            action="store_true",
            help="Clean out any files in the directory first",
        )
        parser.add_argument(
            "--use-hartmann-optimum",
            type=str,
            nargs="+",
            required=False,
            help=(
                "The name of the prior(s) to replace with the optimum if using"
                " hartmann benchmarks. Must be contained in `--priors`"
            ),
        )

        return parser


def main() -> int:
    """The main entry point."""
    prog = "python -m mfpbench"
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    handlers = {handler.name: handler for handler in CommandHandler.get()}
    for name, handler in handlers.items():
        line = "-" * len(handler.help) + "\n"
        subparser = subparsers.add_parser(
            name,
            help=f"{prog} {name} --help\n\n{handler.help}\n{line}",
        )
        handler.fill_parser(subparser)

    args = parser.parse_args()
    _handler = handlers.get(args.command)
    if _handler is None:
        parser.print_help()
        return 1

    _handler.do(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
