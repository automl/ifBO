from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from typing_extensions import override

import pandas as pd
from more_itertools import first_true

logger = logging.getLogger(__name__)

DATAROOT = Path("data")
HERE = Path(__file__).parent.absolute()
REQ_DIR = HERE.parent / "requirements"


@dataclass(frozen=True)  # type: ignore[misc]
class BenchmarkSetup(ABC):
    name: ClassVar[str]
    """The name of the benchmark group."""

    supports_parallel: ClassVar[bool] = False
    """Whether this benchmark supports parallel downloading.

    The download method will be called with a `workers` argument.
    """

    @classmethod
    @abstractmethod
    def download(cls, path: Path, workers: int = 1) -> None:
        """Download the data from the source.

        Args:
            path: The root path to download to.
                Will install to
                path/[name][mfpbench.setup_benchmark.BenchmarkSetup.name]
            workers: The number of workers to use for downloading. This
                can be ignored for benchmarks that do not support parallel.
        """
        ...

    @classmethod
    def default_location(cls) -> Path:
        """Get the default location for the data."""
        return DATAROOT / cls.name

    @classmethod
    def default_requirements_path(cls) -> Path:
        """Get the default location for the data."""
        return REQ_DIR / f"{cls.name}.txt"

    @classmethod
    def install_cmd(cls, requirements_path: Path) -> str:
        """Get the command to install the requirements.

        Args:
            requirements_path: The path to the requirements.txt file.
        """
        return f"python -m pip install -r {requirements_path.absolute()}"

    @classmethod
    def install(cls, requirements_path: Path) -> None:
        """Install the requirements to download the data.

        Args:
            requirements_path: The path to the requirements.txt file.
        """
        cmd = cls.install_cmd(requirements_path)
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)  # noqa: S602

    @classmethod
    def source(cls, name: str) -> type[BenchmarkSetup]:
        """Get all the sources."""
        subclasses = cls.__subclasses__()

        found = first_true(subclasses, pred=lambda x: x.name == name, default=None)
        if found is None:
            names = [subclass.name for subclass in subclasses]
            raise ValueError(f"No source with {name=}\nPlease choose from {names}")

        return found

    @classmethod
    def sources(cls) -> list[type[BenchmarkSetup]]:
        """Get all the sources."""
        return cls.__subclasses__()


@dataclass(frozen=True)
class YAHPOSource(BenchmarkSetup):
    name = "yahpo"
    tag: str = "v1.0"
    git_url: str = "https://github.com/slds-lmu/yahpo_data"

    @override
    @classmethod
    def download(cls, path: Path, workers: int = 1) -> None:
        cmd = f"git clone --depth 1 --branch {cls.tag} {cls.git_url} {path}"
        subprocess.run(cmd.split(), check=True)  # noqa: S603


@dataclass(frozen=True)
class JAHSBenchSource(BenchmarkSetup):
    name = "jahs"
    url = "https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar"

    @override
    @classmethod
    def download(cls, path: Path) -> None:
        tarpath = path / "assembled_surrogates.tar"

        print(f"Download {cls.url}, this might take a while.")
        if not tarpath.exists():
            with urllib.request.urlopen(  # noqa: S310
                cls.url,
            ) as response, tarpath.open("wb") as f:
                shutil.copyfileobj(response, f)

        print("Download finished, extracting now")
        with tarfile.open(tarpath, "r") as f:
            f.extractall(path=path)


@dataclass(frozen=True)
class PD1Source(BenchmarkSetup):
    url: str = "http://storage.googleapis.com/gresearch/pint/pd1.tar.gz"
    surrogate_url: str = (
        "https://ml.informatik.uni-freiburg.de/research-artifacts/mfp-bench"
    )
    surrogate_version: str = "vPaper-Arxiv"
    name = "pd1"

    @override
    @classmethod
    def download(cls, path: Path, workers: int = 1) -> None:
        cls._download_surrogates(path)

    @classmethod
    def _download_rawdata(cls, path: Path) -> None:
        tarpath = path / "rawdata.tar.gz"
        print(f"Downloading raw data from {cls.url}")

        if not tarpath.exists():
            with urllib.request.urlopen(  # noqa: S310
                cls.url,
            ) as response, tarpath.open(
                "wb",
            ) as f:
                shutil.copyfileobj(response, f)

        print(f"Done downloading raw data from {cls.url}")

    @classmethod
    def _download_surrogates(cls, path: Path) -> None:
        surrogate_dir = path / "surrogates"
        surrogate_dir.mkdir(exist_ok=True, parents=True)
        zip_path = surrogate_dir / "surrogates.zip"

        # Download the surrogates zip
        url = f"{cls.surrogate_url}/{cls.surrogate_version}/surrogates.zip"
        print(f"Downloading from {url}")

        if not zip_path.exists():
            with urllib.request.urlopen(url) as response, zip_path.open(  # noqa: S310
                "wb",
            ) as f:
                shutil.copyfileobj(response, f)

        with zipfile.ZipFile(zip_path, "r") as zip:
            zip.extractall(surrogate_dir)

        print(f"Finished downloading from {url}")


@dataclass(frozen=True)
class NB201TabularSource(BenchmarkSetup):
    url: str = "https://ml.informatik.uni-freiburg.de/research-artifacts/mfp-bench"
    tabular_version: str = "v1"
    name = "nb201-tabular"

    @override
    @classmethod
    def download(cls, path: Path, workers: int = 1) -> None:
        zippath = path / "nb201-tabular-data.zip"
        url = f"{cls.url}/{cls.name}/{cls.tabular_version}/nb201-tabular-data.zip"
        if not zippath.exists():
            _urlopen = urllib.request.urlopen
            print(f"Downloading from {url}")
            with _urlopen(url) as response, zippath.open("wb") as f:
                shutil.copyfileobj(response, f)

        with zipfile.ZipFile(zippath, "r") as zip_ref:
            zip_ref.extractall(path)

        print(f"Downloaded {cls.name}")


@dataclass(frozen=True)
class LCBenchTabularSource(BenchmarkSetup):
    url: str = "https://figshare.com/ndownloader/files/21188607"
    name = "lcbench-tabular"

    @override
    @classmethod
    def download(cls, path: Path, workers: int = 1) -> None:
        zippath = path / "data_2k.zip"
        if not zippath.exists():
            _urlopen = urllib.request.urlopen
            print(f"Downloading from {cls.url}")
            with _urlopen(cls.url) as response, zippath.open("wb") as f:
                shutil.copyfileobj(response, f)

        with zipfile.ZipFile(zippath, "r") as zip_ref:
            zip_ref.extractall(path)

        print(f"Downloaded {cls.name}")
        cls._process(path)

    @classmethod
    def _process(cls, path: Path) -> None:
        filepath = path / "data_2k.json"
        print(f"Processing {cls.name} ...")
        with filepath.open("r") as f:
            all_data = json.load(f)

        for dataset_name, data in all_data.items():
            logger.info(f"Processing {dataset_name}")
            config_frames_for_dataset = []
            for config_id, config_data in data.items():
                config: dict = config_data["config"]

                log_data: dict = config_data["log"]
                loss: list[str] = log_data["Train/loss"]  # Name of the loss
                val_ce: list[float] = log_data["Train/val_cross_entropy"]
                val_acc: list[float] = log_data["Train/val_accuracy"]
                val_bal_acc: list[float] = log_data["Train/val_balanced_accuracy"]

                test_ce: list[float] = log_data["Train/test_cross_entropy"]
                test_bal_acc: list[float] = log_data["Train/test_balanced_accuracy"]

                # NOTE: Due to there being a lack of "Test/val_accuracy" in the
                # data but a "Train/test_result" we use the latter as the test accuracy
                test_acc: list[float] = log_data["Train/test_result"]

                time = log_data["time"]
                epoch = log_data["epoch"]

                df = pd.DataFrame(
                    {
                        "time": time,
                        "epoch": epoch,
                        "loss": loss,
                        "val_accuracy": val_acc,
                        "val_cross_entropy": val_ce,
                        "val_balanced_accuracy": val_bal_acc,
                        "test_accuracy": test_acc,
                        "test_cross_entropy": test_ce,
                        "test_balanced_accuracy": test_bal_acc,
                    },
                )
                # These are single valued but this will make them as a list into
                # the dataframe
                df = df.assign(**{"id": config_id, **config})

                config_frames_for_dataset.append(df)

            #                     | **metrics, **config_params
            # (config_id, epoch)  |
            df_for_dataset = (
                pd.concat(config_frames_for_dataset, ignore_index=True)
                .convert_dtypes()
                .set_index(["id", "epoch"])
                .sort_index()
            )
            table_path = path / f"{dataset_name}.parquet"
            df_for_dataset.to_parquet(table_path)
            logger.info(f"Processed {dataset_name} to {table_path}")


class PD1TabularSource(BenchmarkSetup):
    url: str = "http://storage.googleapis.com/gresearch/pint/pd1.tar.gz"
    name = "pd1-tabular"

    @override
    @classmethod
    def download(cls, path: Path, workers: int = 1) -> None:
        zippath = path / "pd1.tar.gz"
        if not zippath.exists():
            _urlopen = urllib.request.urlopen
            print(f"Downloading from {cls.url}")
            with _urlopen(cls.url) as response, zippath.open("wb") as f:
                shutil.copyfileobj(response, f)
        print(f"Downloaded {cls.name}")
        cls._process(zippath)

    @classmethod
    def _process(cls, path: Path) -> None:
        from mfpbench.pd1.processing.process_script import process_pd1

        process_pd1(path, process_tabular=True)


class TaskSetabularSource(BenchmarkSetup):
    name = "taskset-tabular"
    supports_parallel = True

    @override
    @classmethod
    def download(cls, path: Path, workers: int = 1) -> None:
        cls._process(path, workers=workers)

    @classmethod
    def _process(cls, path: Path, workers: int = 1) -> None:
        from mfpbench.taskset_tabular.processing.process import process_taskset

        process_taskset(output_dir=path, workers=workers)


def download_status(source: str, datadir: Path | None = None) -> bool:
    """Check whether the data is downloaded for some source."""
    datadir = datadir if datadir is not None else DATAROOT
    _source = BenchmarkSetup.source(source)
    source_path = datadir / _source.name
    return source_path.exists() and bool(
        next(source_path.iterdir(), False),
    )


def print_download_status(
    sources: list[str] | None = None,
    datadir: Path | None = None,
) -> None:
    """Print the status of the data.

    Args:
        sources: The benchmarks to check the status of. `None` for all.
        datadir: Where the root data directory is
    """
    datadir = datadir if datadir is not None else DATAROOT
    s = f"root: {datadir.absolute()}"
    print(s)
    print("-" * len(s))

    if (sources is not None and "all" in sources) or sources is None:
        names = [source.name for source in BenchmarkSetup.sources()]
    else:
        names = sources

    for name in names:
        if download_status(name, datadir=datadir):
            print(f"[✓] {name}")
        else:
            print(f"[x] {name: <20} python -m mfpbench download --benchmark {name}")


def print_requirements(benchmarks: list[str]) -> None:
    """Print the status of the data.

    Args:
        benchmarks: The benchmarks to check the status of. `None` for all.
    """
    sources = BenchmarkSetup.sources()
    if benchmarks is not None and "all" not in benchmarks:
        sources = [source for source in sources if source.name in benchmarks]

    for source in sources:
        print("=" * len(source.name))
        print(f"{source.name}")
        print("=" * len(source.name))

        path = source.default_requirements_path()
        pathstr = f"path: {path}"
        cmd = source.install_cmd(path)
        cmdstr = f"cmd: {cmd}"
        execpath = sys.executable
        execstr = f"exec: {execpath}"
        n = max(len(pathstr), len(cmdstr), len(execstr))

        print(pathstr)
        print(execstr)
        print(cmdstr)
        print("-" * n)
        if not path.exists():
            print("Not found!")
        else:
            print(f"# {path}")
            with path.open("r") as f:
                print(f.read())
        print()


def setup(
    benchmark: str,
    *,
    datadir: Path | None = None,
    download: bool = True,
    install: str | bool = False,
    force: bool = False,
    workers: int = 1,
) -> None:
    """Download data for a benchmark.

    Args:
        benchmark: The benchmark to download the data for.
        datadir: Where the root data directory is
        download: Whether to download the data
        install: Whether to install the requirements for the benchmark.
            If True, will install the default. If a str, tries to interpret
            it as a full path.
        force: Whether to force redownload of the data
        workers: The number of workers to use for downloading. This
            will be ignored for benchmarks that do not support parallel
            setup.
    """
    datadir = datadir if datadir is not None else DATAROOT

    source = BenchmarkSetup.source(benchmark)
    source_path = datadir / source.name

    if download:
        if source_path.exists() and force:
            print(f"Removing {source_path}")
            shutil.rmtree(source_path)

        if not source_path.exists() or next(source_path.iterdir(), None) is None:
            print(f"Downloading to {source_path}")
            source_path.mkdir(exist_ok=True, parents=True)
            source.download(source_path, workers=workers)
            print(f"Finished downloading to {source_path}")
        else:
            print(f"Already found something at {source_path}")

    if install is not False:
        if install is True:
            req_path = source.default_requirements_path()
        else:
            req_path = Path(install)
            if not req_path.exists():
                raise FileNotFoundError(f"Could not find requirements at {req_path}")

        print(f"Installing requirements at {req_path}")
        source.install(req_path)


if __name__ == "__main__":
    LCBenchTabularSource._process(Path("data/lcbench-tabular"))
