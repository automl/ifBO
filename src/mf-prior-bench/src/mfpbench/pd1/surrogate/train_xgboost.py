from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

# TODO: Should really move this
from mfpbench.pd1.processing.columns import COLUMNS

if TYPE_CHECKING:
    from ConfigSpace import Configuration
    from xgboost import XGBRegressor

HERE = Path(__file__).absolute().resolve().parent
DATADIR = HERE.parent.parent.parent / "data"


def train_xgboost(
    config: Configuration,
    budget: int,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int | None = None,
) -> XGBRegressor:
    """Train an XGBoost model on the given data.

    Args:
        config: The configuration to use for the XGBoost model
        budget: The number of estimators to use for the XGBoost model
        X: The data to train on
        y: The target to train on
        seed: The seed to use for the XGBoost model

    Returns:
        The trained XGBoost model
    """
    from xgboost import XGBRegressor

    if y.name == "train_cost":
        model = XGBRegressor(
            **config,
            seed=seed,
            n_estimators=budget,
            monotone_constraints={"epoch": 1},
        )
    else:
        model = XGBRegressor(**config, seed=seed, n_estimators=budget)

    model.fit(X, y)

    return model  # type: ignore


if __name__ == "__main__":
    import argparse

    from xgboost import XGBRegressor

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--to", required=True, type=str)

    parser.add_argument(
        "--y",
        choices=["valid_error_rate", "test_error_rate", "train_cost"],
        required=True,
        type=str,
    )

    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--booster", type=str, required=True)
    parser.add_argument("--colsample_bytree", type=float, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--max_depth", type=int, required=True)
    parser.add_argument("--reg_lambda", type=float, required=True)
    parser.add_argument("--subsample", type=float, required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.y not in df.columns:
        raise ValueError(f"Can't train for {args.y} for dataset {args.dataset}")

    metrics = [c.rename if c.rename else c.name for c in COLUMNS if c.metric]
    valid_metrics = [m for m in metrics if m in df.columns]

    df = df.dropna()
    X = df.drop(columns=valid_metrics)
    y = df[args.y]

    config = {
        "alpha": args.alpha,
        "booster": args.booster,
        "colsample_bytree": args.colsample_bytree,
        "eta": args.eta,
        "max_depth": args.max_depth,
        "reg_lambda": args.reg_lambda,
        "subsample": args.subsample,
    }

    xgboost_model = train_xgboost(
        X=X,
        y=y,
        seed=args.seed,
        budget=args.budget,
        config=config,
    )

    xgboost_model.save_model(args.to)

    # Try load it?
    loaded_model = XGBRegressor()
    loaded_model.load_model(args.to)

    y_pred = loaded_model.predict(X)
