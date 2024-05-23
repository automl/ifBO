from __future__ import annotations

import gzip
import json
import logging
import shutil
import warnings
from dataclasses import dataclass
from itertools import accumulate, product
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from mfpbench.pd1.processing.columns import COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# HPS to get from certain PD1 benchmarks
HPS = ["lr_decay_factor", "lr_initial", "lr_power", "opt_momentum"]

# Benchmarks we do not process and discard
BAD_BENCHMARKS = [
    "fashion_mnist-max_pooling_cnn-256_tabular.csv",
    "fashion_mnist-max_pooling_cnn-2048_tabular.csv",
    "mnist-max_pooling_cnn-256_tabular.csv",
    "mnist-max_pooling_cnn-2048_tabular.csv",
]

NUM_FID_STEP_THRESHOLD_HIGH = 100  # maximum steps allowed to not subsample
NUM_FID_STEP_THRESHOLD_LOW = 10  # minimum steps required to be a benchmark
FID_SUBSAMPLING_STEP_JUMPS = [1, 2, 5, 10]


def safe_accumulate(
    x: Iterator[float | None] | float,
    fill: float = np.nan,
) -> Iterator[float]:
    """Accumulate, but fill in missing values with a default value."""
    if isinstance(x, float):
        return iter([fill if np.isnan(x) else x])

    itr = iter(f if f is not None else fill for f in x)
    return accumulate(itr)


def uniref50_epoch_convert(x: float | list[float]) -> float | list[float]:
    """Converts the epochs of uniref50 to some usable form.

    Converts:
        0             NaN
        1    [0, 0, 0, 1]
        2           [nan]
        3     [0, 0, nan].

    to:
        0             NaN
        1    [0, 1, 2, 3]
        2           [nan]
        3     [0, 1, nan]
    """
    if isinstance(x, list):
        return [i if not pd.isna(e) else e for i, e in enumerate(x, start=1)]

    return x


@dataclass(frozen=True)
class Datapack:
    matched: bool
    phase: int
    dir: Path

    @property
    def _rawname(self) -> str:
        m = "matched" if self.matched else "unmatched"
        return f"pd1_{m}_phase{self.phase}_results"

    @property
    def _archive_path(self) -> Path:
        fname = self._rawname + ".jsonl.gz"
        return self.dir / fname

    def _unpack(self) -> pd.DataFrame:
        frm = self._archive_path

        if not frm.exists():
            raise FileNotFoundError(f"No archive found at {frm}")

        if not self.unpacked_path.exists():
            logger.info(f"Unpacking from {frm}")
            with gzip.open(frm, mode="rt") as f:
                data = [json.loads(line) for line in f]
            unpacked = pd.DataFrame(data)

            logger.info(f"Saving to {self.unpacked_path}")
            unpacked.to_csv(
                self.unpacked_path,
                index=False,
            )
        else:
            logger.info(f"Unpacking from {frm}")
            unpacked = pd.read_csv(self.unpacked_path)
            assert isinstance(unpacked, pd.DataFrame)
            columns = {col.name: col for col in COLUMNS}

            # Convert a string representing a list to a
            # real list of the contained objects using json.loads
            list_columns = [
                col
                for col in unpacked.columns
                if col in columns and columns[col].type is list
            ]
            for c in list_columns:
                unpacked[c] = [
                    json.loads(val.replace("None", "null"))
                    if isinstance(val, str)
                    else val
                    for _, val in unpacked[c].items()  # type: ignore
                ]

            assert isinstance(unpacked, pd.DataFrame)

        return unpacked

    @property
    def unpacked_path(self) -> Path:
        """The path to the unpacked csv."""
        return self.dir / f"{self._rawname}_unpacked.csv"


def process_pd1(  # noqa: C901, PLR0912, PLR0915
    tarball: Path,
    *,
    process_tabular: bool = False,
) -> None:
    """Process the pd1 dataset.

    !!! note

        Will write to the same directory as the tarball

    Args:
        tarball: The path to the tarball containing the raw data.
        process_tabular: Whether to process the data for tabular benchmarking.
    """
    datadir = tarball.parent
    rawdir = tarball.parent / "raw"

    fulltable_name = "full.csv"
    fulltable_path = datadir / fulltable_name

    # If we have the full table we can skip the tarball extraction
    if not fulltable_path.exists():
        if not tarball.exists():
            raise FileNotFoundError(f"No tarball found at {tarball}")

        rawdir.mkdir(exist_ok=True)

        # Unpack it to the rawdir
        readme_path = rawdir / "README.txt"
        if not readme_path.exists():
            shutil.unpack_archive(tarball, rawdir)

            unpacked_folder_name = "pd1"  # This is what the tarball will unpack into
            unpacked_folder = rawdir / unpacked_folder_name

            # Move everything from the uncpack folder to the "raw" folder
            for filepath in unpacked_folder.iterdir():
                to = rawdir / filepath.name
                shutil.move(str(filepath), str(to))

            # Remove the archive folder, its all been moved to "raw"
            shutil.rmtree(str(unpacked_folder))

    # For processing the df
    drop_columns = [c.name for c in COLUMNS if not c.keep]
    renames = {c.name: c.rename for c in COLUMNS if c.rename is not None}
    hps = [c.rename for c in COLUMNS if c.hp]
    # metrics = [c.rename if c.rename else c.name for c in COLUMNS if c.metric]

    dfs: list[pd.DataFrame] = []
    for matched, phase in product([True, False], [0, 1]):
        # Unpack the jsonl.gz archive if needed
        datapack = Datapack(matched=matched, phase=phase, dir=rawdir)
        df = datapack._unpack()

        # Tag them from the dataset they came from
        df["matched"] = matched
        df["phase"] = phase

        df = df.drop(columns=drop_columns)
        df = df.rename(columns=renames)

        dfs.append(df)

    # We now merge them all into one super table for convenience
    full_df = pd.concat(dfs, ignore_index=True)

    # Since some columns values are essentially lists, we need to explode them out
    # However, we need to make sure to distuinguish between transformer and not as
    # transformers do not have test error available
    # We've already renamed the columns them at this point
    list_columns = [c.rename if c.rename else c.name for c in COLUMNS if c.type == list]
    transformer_datasets = ["uniref50", "translate_wmt", "imagenet", "lm1b"]
    dataset_columns = ["dataset", "model", "batch_size"]

    groups = full_df.groupby(dataset_columns)
    for (name, model, batchsize), _dataset in groups:  # type: ignore
        fname = f"{name}-{model}-{batchsize}"
        logger.info(fname)

        if name in transformer_datasets:
            explode_columns = [c for c in list_columns if c != "test_error_rate"]
            dataset = _dataset.drop(columns=["test_error_rate"])
        else:
            explode_columns = list_columns
            dataset = _dataset

        if name == "uniref50" and process_tabular is False:
            # For some reason the epochs of this datasets are basically [0, 0, 0, 1]
            # We just turn this into an incremental thing
            epochs = dataset["epoch"]
            assert epochs is not None
            dataset["epoch"] = dataset["epoch"].apply(  # type: ignore
                uniref50_epoch_convert,
            )

        # Make sure train_cost rows are all of equal length
        dataset["train_cost"] = [
            np.nan if r in (None, np.nan) else list(safe_accumulate(r, fill=np.nan))
            for r in dataset["train_cost"]  # type: ignore
        ]

        # Explode out the lists in the entries of the datamframe to be a single long
        # dataframe with each element of that list on its own row
        dataset = dataset.explode(explode_columns, ignore_index=True)
        logger.info(f"{len(dataset)} rows")
        assert isinstance(dataset, pd.DataFrame)

        # Remove any rows that have a nan as cost in the exploded columns
        nan_rows = dataset["train_cost"].isna()  # for PD1 if cost is NaN so is `epoch`
        logger.info(f" - len(nan_rows) {sum(nan_rows)}")

        logger.debug(f"Removing rows with nan in {explode_columns}")
        dataset = dataset[~nan_rows]  # type: ignore
        assert isinstance(dataset, pd.DataFrame)

        logger.info(f"{len(dataset)} rows (after nan removal)")

        if fname == "lm1b-transformer-2048" and process_tabular is False:
            # Some train costs go obscenely high for no reason, we drop these rows
            dataset = dataset[dataset["train_cost"] < 10_000]  # type: ignore

        elif fname == "uniref50-transformer-128" and process_tabular is False:
            # Some train costs go obscenely high for no reason, we drop these rows
            # Almost all are below 400 but we add a buffer
            dataset = dataset[dataset["train_cost"] < 4_000]  # type: ignore

        elif fname == "imagenet-resnet-512" and process_tabular is False:
            # We drop all configs that exceed the 0.95 quantile in their max train_cost
            # as we consider this to be a diverging config. The surrogate will smooth
            # out these configs as it is not aware of divergence
            # NOTE: q95 was experimentally determined so as to not remove too many
            # configs but remove configs which would create massive gaps in "train_cost"
            # which would cause optimization of the surrogate to focus too much on
            # minimizing it's loss for outliers
            hp_names = HPS
            maxes = [
                v["train_cost"].max()  # type: ignore
                for _, v in dataset.groupby(hp_names)
            ]
            q95 = np.quantile(maxes, 0.95)
            configs_who_dont_exceed_q95 = (
                v
                for _, v in dataset.groupby(hp_names)
                if v["train_cost"].max() < q95  # type: ignore
            )
            dataset = pd.concat(configs_who_dont_exceed_q95, axis=0)

        elif fname == "cifar100-wide_resnet-2048" and process_tabular is False:
            # We drop all configs that exceed the 0.93 quantile in their max train_cost
            # as we consider this to be a diverging config. The surrogate will smooth
            # out these configs as it is not aware of divergence
            # NOTE: q93 was experimentally determined so as to not remove too many
            # configs but remove configs which would create massive gaps in
            # "train_cost" which would cause optimization of the surrogate to
            # focus too much on minimizing it's loss for outliers
            hp_names = HPS
            maxes = [
                v["train_cost"].max()  # type: ignore
                for _, v in dataset.groupby(hp_names)
            ]
            q93 = np.quantile(maxes, 0.93)
            configs_who_dont_exceed_q93 = (
                v
                for _, v in dataset.groupby(hp_names)
                if v["train_cost"].max() < q93  # type: ignore
            )
            dataset = pd.concat(configs_who_dont_exceed_q93, axis=0)

        # We want to write the full mixed {phase,matched} for surrogate training while
        # only keeping matched phase 1 data for tabular.
        # We also no longer need to keep dataset, model and batchsize for individual
        # datasets.
        # We can also drop "activate_fn" for all but 4 datasets
        has_activation_fn = [
            "fashion_mnist-max_pooling_cnn-256",
            "fashion_mnist-max_pooling_cnn-2048",
            "mnist-max_pooling_cnn-256",
            "mnist-max_pooling_cnn-2048",
        ]
        drop_columns = ["dataset", "model", "batch_size"]
        if process_tabular or fname not in has_activation_fn:
            drop_columns += ["activation_fn"]

        dataset = dataset.drop(columns=drop_columns)  # type: ignore

        # Select only the tabular part (matched and phase1)
        if process_tabular:
            tabular_path = datadir / f"{fname}_tabular.csv"
            if fname == "imagenet-resnet-1024":
                tabular_mask = ~dataset["matched"] & (dataset["phase"] == 0)
            else:
                tabular_mask = dataset["matched"] & (dataset["phase"] == 1)
            df_tabular = dataset[tabular_mask]
            df_tabular = df_tabular.drop(columns=["matched", "phase"])  # type: ignore

            # print(f"Writing tabular data to {tabular_path}")
            df_tabular.to_csv(tabular_path, index=False)
            # TODO: Not sure why we dont just pass the actual table
            _data = preprocess_csv_for_tabular_benchmark_dfs(tabular_path)
        else:
            # There are some entries which seem to appear twice. This is due to the same
            # config in {phase0,phase1} x {matched, unmatched}
            # To prevent issues, we simply drop duplicates
            hps = HPS
            hps = [*hps, "activation_fn"] if fname in has_activation_fn else list(hps)

            dataset = dataset.drop_duplicates([*hps, "epoch"], keep="last")  # type: ignore

            # The rest can be used for surrogate training
            surrogate_path = datadir / f"{fname}_surrogate.csv"
            df_surrogate = dataset.drop(columns=["matched", "phase"])
            df_surrogate.to_csv(surrogate_path, index=False)
    # end of for


# end of process_pd1()


def preprocess_csv_for_tabular_benchmark_dfs(path: Path) -> None:
    assert path.is_file(), "Need a valid file path, not a directory!"

    if not path.name.endswith("_tabular.csv"):
        return

    if str(path.name) in BAD_BENCHMARKS:
        warnings.warn(f"Discarding benchmark: {path.name}", stacklevel=2)
        return

    df = pd.read_csv(path, float_precision="high")

    # find the unique set of hyperparameters/configs
    # There are duplicates of the HP columns because they appear multiple times,
    # one occurence for each epoch they were evaluated at.
    #       HP1 | HP2 | HP3 | HP4
    # 0   |
    # 126 |
    # 252 |
    # ...
    unique_df = df.loc[:, HPS].drop_duplicates(keep="first").sort_index()

    # assigning a unique number to each config, making sure that the index
    # matches to the unique_df index
    #         "id"
    # 0     | 1
    # 126   | 2
    # 252   | 3
    # ...
    config_ids = pd.Series(
        np.arange(1, len(unique_df.index) + 1),
        index=unique_df.index,
    )

    # Assign the new config_ids to the original dataframe
    #       HPs...  | epoch     | Results...    | id
    # 0   |             0                       | 1.0
    # 0   |             1                       | nan
    # 0   |             2                       | nan
    # ...
    # 126 |             0                       | 2.0
    # 126 |             1                       | nan
    # 126 |             2                       | nan
    # ...
    # 252 |             0                       | 3.0
    # 252 |             1                       | nan
    # 252 |             2                       | nan
    # ...
    df["id"] = config_ids

    # We forward fill the id column so that each row has a unique id
    df.id = df["id"].ffill().astype(int)

    # calculating total number of steps per unique config
    # TODO: All of this could be made much simpler with a pivot table
    # with specific aggregations, i.e. listing value columsn and having a count
    # for the number of epochs
    fid_steps = (
        df.id.drop_duplicates(keep="last").index.values
        - df.id.drop_duplicates(keep="first").index.values
    )

    # removing all rows that have recorded fewer fidelities than the max frequency seen
    uniques, counts = np.unique(fid_steps, return_counts=True)
    to_remove = np.where(fid_steps != uniques[counts.argmax()])[0]  # list of indices
    idx_to_remove = df.id[unique_df.index.to_numpy()[to_remove]].to_numpy()
    df.index = df.id
    df = df.drop(index=idx_to_remove)

    # re-enumerate indexes for unique configs
    unique_df = df.loc[:, HPS].drop_duplicates(keep="first").sort_index()
    df["id"] = pd.Series(np.arange(1, len(unique_df.index) + 1), index=unique_df.index)
    df.id = df["id"].ffill()
    df.id = df.id.astype(int)

    # retaining original table's epochs
    df["original_steps"] = df.epoch

    # enumerating all fidelities seen
    fid_len = len(df.loc[df.index.values[0]].epoch.values)
    enumerated_fidelities = np.arange(1, fid_len + 1)
    enumerated_fid_col = enumerated_fidelities.tolist() * len(df.index.unique())
    df["epoch"] = enumerated_fid_col

    def is_sub_epoch():
        return len(df.loc[1].original_steps) != len(df.loc[1].original_steps.unique())

    def is_large_num_steps():
        return len(df.loc[1].epoch) > NUM_FID_STEP_THRESHOLD_HIGH

    logger.info(
        f"{path.name}; sub_epoch={is_sub_epoch()}; large_steps={is_large_num_steps()}",
    )

    if is_large_num_steps():
        subsample_steps(df, path)
    else:
        df = df.set_index(["id", "epoch"])
        # Save to disk
        df.to_parquet(path.resolve().parent / f"{path.name.split('.csv')[0]}.parquet")

    return


def subsample_steps(df: pd.DataFrame, path: Path) -> None:
    df_backup = df.copy()
    # subsamples for different step sizes and saves a version of the benchmark
    for jump_step in FID_SUBSAMPLING_STEP_JUMPS:
        df = df_backup.copy()
        target_path = (
            path.resolve().parent / f"{path.name.split('.csv')[0]}-{jump_step}.parquet"
        )
        if jump_step == 1:
            df = df.set_index(["id", "epoch"])
            # Save to disk
            df.to_parquet(target_path)
            continue

        _unique_fids = df.loc[1].epoch.values
        _retain_list, _ = np.linspace(
            start=1,
            stop=_unique_fids[-1],
            num=(len(_unique_fids) - 1) // jump_step,
            retstep=jump_step,
            endpoint=True,
            dtype=int,
        )
        if len(_retain_list) < NUM_FID_STEP_THRESHOLD_LOW:
            print(f"\nNot subsampling {path.name} for jump_step={jump_step}\n‚")
            continue
        drop_list = list(set(_unique_fids) - set(_retain_list))
        df.loc[df["epoch"].isin(drop_list), "epoch"] = np.nan
        df = df.dropna()

        # reindexing
        df = df.set_index(["id", "epoch"])
        # enumerating fidelities again
        df.index = df.index.set_levels(
            np.arange(1, len(df.index.get_level_values(1)) + 1, dtype=int).tolist(),
            level=1,
        )
        # Save to disk
        df.to_parquet(target_path)


if __name__ == "__main__":
    import argparse

    HERE = Path(__file__).resolve().absolute().parent
    DATADIR = HERE.parent.parent.parent / "data" / "pd1-data"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATADIR,
        help="Where the data directory is",
    )
    parser.add_argument(
        "--process_for_tabular",
        action="store_true",
        help="Preprocesses into MultiIndex DataFrame for tabular querying",
    )
    args = parser.parse_args()

    # Print class names
    for f in args.data_dir.iterdir():
        if (
            f.suffix == ".csv"
            and "_matched" in str(f)
            and "phase1" in str(f)
            and not str(f).startswith("pd1")
        ):
            dataset, model, rest = str(f).split("-")
            batchsize, *_ = rest.split("_")
            dataset = dataset.replace("_", " ").title().replace(" ", "")
            model = model.replace("_", " ").title().replace(" ", "")

    tarball = args.data_dir / "data.tar.gz"
    process_pd1(tarball, process_tabular=args.process_for_tabular)

    print("Processed benchmarks!")  # noqa: T201
