import os
import torch
from ifbo import Curve, PredictionResult
from ifbo.utils import tokenize
from pathlib import Path
from typing import List
from .download import download_and_decompress, VERSION_MAP, FILENAME, WEIGHTS_FINAL_NAME


class FTPFN(torch.nn.Module):
    def __init__(self, version: str = "0.0.1"):
        super(FTPFN, self).__init__()
        trained_models_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "trained_models")
        )

        if version not in VERSION_MAP:
            raise ValueError(f"Version {version} is not available")

        dest_path = Path(trained_models_path) / FILENAME(version)

        if not dest_path.exists():
            download_and_decompress(
                url=VERSION_MAP.get(version).get("url"), path=dest_path, version=version
            )

        self.model = torch.load(
            os.path.join(trained_models_path, WEIGHTS_FINAL_NAME(version)),
            map_location="cpu",
        )
        self.model.eval()

    @torch.no_grad()
    def predict(
        self, context: List[Curve], query: List[Curve]
    ) -> List[PredictionResult]:
        x_train, y_train, x_test = tokenize(context, query)
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        results = torch.split(logits, [len(curve.t) for curve in query], dim=0)
        return [
            PredictionResult(
                logits=logit,
                criterion=self.model.criterion,
            )
            for curve, logit in zip(query, results)
        ]

    def check_input(self, x_train, y_train, x_test):
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

    def forward(self, x_train, y_train, x_test):
        self.check_input(x_train, y_train, x_test)
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
