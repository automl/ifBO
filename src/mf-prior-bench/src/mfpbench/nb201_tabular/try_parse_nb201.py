"""This script parsers out results from NAS-Bench-201 and
saves them as parquet files.
"""
# They mention in API doc that ori-test for `cifar10-valid` is only on test set
# while for the others, it means val+test set
# https://github.com/D-X-Y/NAS-Bench-201/blob/8558547969c131f75af2725869ff1ece98e98f23/nas_201_api/api_utils.py#L361
# NOTE: acc is on a scale of 0 to 100
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from nas_201_api import NASBench201API as API

CACHED = Path(".cached-nb201")
if not CACHED.exists():
    print("no pickled")
    api = API("TODO")
    with CACHED.open("wb") as f:
        import pickle

        pickle.dump(api, f)
    print("loaded")
    print("pickled")
else:
    import pickle

    print("from pickled")

    with CACHED.open("rb") as f:
        api = pickle.load(f)

    print("loaded")

# api.verbose = False
EPOCHS = [12, 200]


def column_remapper(dataset: str) -> Callable[[str], str]:
    # Takes place after dropping
    # Here we basically rename whatever evaluation curve we have that wasn't
    # `train` and set it's name to `test`
    if dataset == "cifar10-valid":
        return lambda x: x.replace("-", "_").replace("valid", "test")
    if dataset == "cifar10":
        return lambda x: x.replace("-", "_")
    if dataset == "ImageNet16-120":
        return lambda x: x.replace("-", "_").replace("valtest", "test")
    if dataset == "cifar100":
        return lambda x: x.replace("-", "_").replace("valtest", "test")

    raise ValueError(f"Unknown dataset {dataset}")


def columns_to_drop(dataset: str) -> list[str]:
    keys = ["loss", "accuracy", "per-time", "all-time"]
    if dataset == "cifar10-valid":
        # Has `train` and `valid` over entire learning curve
        # Has `test` extra at last epoch
        return [f"test-{k}" for k in keys]
    if dataset == "cifar10":
        # Just has `train` and `test` over entire learning curve
        # Nothing extra at last epoch
        return []
    if dataset == "ImageNet16-120":
        # Has `train` and `valtest` over entire learning curve
        # Has `test` and `valid` extra at last epoch
        return [f"valid-{k}" for k in keys] + [f"test-{k}" for k in keys]
    if dataset == "cifar100":
        # Has `train` and `valtest` over entire learning curve
        # Has `valid` and `test` extra at last epoch
        return [f"valid-{k}" for k in keys] + [f"test-{k}" for k in keys]

    raise ValueError(f"Unknown dataset {dataset}")


def parse_architecture_string(s: str) -> dict[str, str]:
    nodes = s.split("+")
    items = {}
    for node_number, n in enumerate(nodes, start=1):
        # Strip away the `|` at the start and end and the split on `|` in the middle
        # The `:-2` removes the ~<number> indicating which edge it came from.
        # These are ordered anyways so we don't need it
        connections = [c[:-2] for c in n.strip("|").split("|")]
        for node_from, op in enumerate(connections, start=0):
            items[f"edge_{node_from}_{node_number}"] = op
    return items


def arch_to_table(
    arch: int,  # Use as config id, made sure that all arch strs are unique
    dataset_name: str,
    max_epoch: int,
) -> pd.DataFrame:
    arch_str: str = api.arch(arch)
    arch_as_hps: dict[str, str] = parse_architecture_string(arch_str)
    config_id = arch
    return (
        pd.DataFrame.from_records(
            {
                "epoch": epoch
                + 1,  # We add 1 to make it from 1 to max_epoch (inclusive)
                **api.get_more_info(
                    index=arch,
                    dataset=dataset_name,
                    iepoch=epoch,
                    hp=str(
                        max_epoch,
                    ),  # Yup, weird naming on their behalf, it's either ['12', '200']  # noqa: E501
                    is_random=False,  # This cases them to mean the different seeds
                ),
            }
            for epoch in range(max_epoch)  # Not inclusive as 0 is their first
        )
        .assign(config_id=config_id, **arch_as_hps)
        .astype({"config_id": int, "epoch": int})
        .set_index(["config_id", "epoch"])
    )


def table_for_dataset(
    dataset_name: str,
    max_epoch: int,
    archs: list[int],
    cached: bool = True,
) -> pd.DataFrame:
    path = Path(f"nb201_{dataset_name}_{max_epoch}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    _df = pd.concat([arch_to_table(arch, dataset_name, max_epoch) for arch in archs])
    hps = _df.columns[_df.columns.str.startswith("edge_")]
    _df = _df.astype({hp: "category" for hp in hps})
    _df = _df.convert_dtypes()
    _df.to_parquet(path)
    return _df


archs = list(range(15625))
for dataset_name in ["cifar10", "cifar10-valid", "cifar100", "ImageNet16-120"]:
    for max_epoch in [12, 200]:
        try:
            print(dataset_name, max_epoch)
            df = table_for_dataset(dataset_name, max_epoch, archs)
            hps = df.columns[df.columns.str.startswith("edge_")]
            df = df.astype({hp: "category" for hp in hps})
            df = df.drop(
                columns=columns_to_drop(dataset_name),
                errors="raise",
            )
            df = df.rename(mapper=column_remapper(dataset_name), axis="columns")

            # Config 9075 for ImageNet w/ 200 epochs is missing train and test loss from
            # epoch 68 onwards. It was the only config of all of nb201 with missing
            # data so we drop it.
            if dataset_name == "ImageNet16-120" and max_epoch == 200:
                df = df.drop(index=(9075,))

            if df.isna().any().any():
                raise ValueError(f"NaNs found in {dataset_name} {max_epoch}")

            df.to_parquet(f"nb201_{dataset_name}_{max_epoch}.parquet")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(e)
