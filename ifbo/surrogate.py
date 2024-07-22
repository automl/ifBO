from __future__ import annotations

import os
from pathlib import Path
import warnings

import torch

from ifbo.download import download_and_decompress
from ifbo.download import FILE_URL
from ifbo.download import FILENAME
from ifbo.download import VERSION_MAP
from ifbo.download import WEIGHTS_FINAL_NAME
from ifbo.utils import Curve
from ifbo.utils import PredictionResult
from ifbo.utils import tokenize


def _resolve_model_path(target_path: Path, version: str) -> Path:
    """Resolve the model path.

    Args:
        target_path (Path): Path to the trained model.
        version (str): Version of the model.

    Returns:
        path: Path to the trained model.
    """
    # Resolve target path
    if target_path is None:
        target_path = Path.cwd().absolute() / ".model"
        warnings.warn(
            "No target path provided. "
            f"Saving the model in the current working directory: {target_path}"
        )
    else:
        if target_path.name == ".model" and target_path.is_dir():
            target_path = target_path.absolute()
        elif (target_path / ".model").is_dir() or (
            target_path.is_dir() and not (target_path / ".model").is_dir()
        ):
            # if target_path is a directory, and if `.model` subdirectory exists or not
            target_path = (target_path / ".model").absolute()
        else:
            raise ValueError("Invalid target path. Please provide a valid directory path.")
    target_path.mkdir(parents=True, exist_ok=True)

    return target_path


class FTPFN(torch.nn.Module):
    """FTPFN surrogate model."""

    def __init__(
        self,
        target_path: Path,
        version: str = "0.0.1",
        device: torch.device | None = None,
    ):
        """Initialize the FTPFN surrogate model.

        Args:
            target_path (Path, optional): Path to the trained model. Defaults to None.
                If None, creates a `.model/` directory in the current working directory.
            version (str, optional): Version of the model. Defaults to "0.0.1".
        """
        super(FTPFN, self).__init__()

        self.version = version
        self.target_path = _resolve_model_path(target_path, self.version)
        self.device = device

        if self.version not in VERSION_MAP:
            raise ValueError(f"Version {version} is not available")

        _target_file_zip = self.target_path / FILENAME(self.version)
        if not download_and_decompress(
            url=FILE_URL(self.version), path=_target_file_zip, version=version
        ):
            raise ValueError(f"Failed to download and decompress the file at {self.target_path}!")

        # Loading and initializing the model with the pre-trained weights
        self.model = torch.load(
            os.path.join(self.target_path, WEIGHTS_FINAL_NAME(version)),
            map_location=self.device if self.device is not None else torch.device("cpu"),
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, context: list[Curve], query: list[Curve]) -> list[PredictionResult]:
        """Obtain the logits for the given context and query curves.

        Function to perform Bayesian inference using FT-PFN that uses the logits obtained to
        compute various measures like likelihood, UCB, EI, PI, and quantile.
        """
        x_train, y_train, x_test = tokenize(context, query, device=self.device)
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        results = torch.split(logits, [len(curve.t) for curve in query], dim=0)
        return [
            PredictionResult(
                logits=logit,
                criterion=self.model.criterion,
            )
            for curve, logit in zip(query, results)
        ]

    def _check_input(
        self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor
    ) -> None:
        if y_train.min() < 0 or y_train.max() > 1:
            raise Exception("y values should be in the range [0,1]")
        if (
            x_train[:, 1].min() < 0
            or x_train[:, 1].max() > 1
            or x_test[:, 1].min() < 0
            or x_test[:, 1].max() > 1
        ):
            raise Exception("step values should be in the range [0,1]")
        if (
            x_train[:, 0].min() < 0
            or x_train[:, 0].max() > 1000
            or x_test[:, 0].min() < 0
            or x_test[:, 0].max() > 1000
        ):
            raise Exception("id values should be in the range [0,1000]")
        if (
            x_train[:, 2:].min() < 0
            or x_train[:, 2:].max() > 1
            or x_test[:, 2:].min() < 0
            or x_test[:, 2:].max() > 1
        ):
            raise Exception("hyperparameter values should be in the range [0,1]")

    def forward(
        self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model."""
        self._check_input(x_train, y_train, x_test)
        if x_train.shape[0] == 0:
            x_test[:, 0] = 0
        elif x_train[:, 0].min() == 0:
            x_train[:, 0] += 1
            x_test[:, 0] += 1

            # reserve id=0 to curves that are not in x_train
            # set to 0 for all id in x_test[:, 0] that is not x_train[:, 0]
            x_test[:, 0] = torch.where(
                torch.isin(x_test[:, 0], x_train[:, 0]),
                x_test[:, 0],
                torch.zeros_like(x_test[:, 0]),
            )

        single_eval_pos = x_train.shape[0]
        batch_size = 2000  # maximum batch size
        n_batches = (x_test.shape[0] + batch_size - 1) // batch_size

        results = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, x_test.shape[0])
            x_batch = torch.cat([x_train, x_test[start:end]], dim=0).unsqueeze(1)
            y_batch = y_train.unsqueeze(1)
            result = self.model((x_batch, y_batch), single_eval_pos=single_eval_pos)
            results.append(result)

        return torch.cat(results, dim=0)
