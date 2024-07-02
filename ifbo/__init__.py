import os
from dataclasses import dataclass
from typing import Optional

import torch

from .download import VERSION_MAP
from .model import Surrogate
from .bar_distribution import BarDistribution

@dataclass
class Curve:
    hyperparameters: torch.Tensor
    t: torch.Tensor
    y: Optional[torch.Tensor] = None

@dataclass(unsafe_hash=True)
class PredictionResult:
    hyperparameters: torch.Tensor
    t: torch.Tensor
    logits: torch.Tensor
    criterion: BarDistribution

    @torch.no_grad()
    def likelihood(self, y_test):
        return -self.criterion(self.logits, y_test)

    @torch.no_grad()
    def ucb(self):
        return self.criterion.ucb(self.logits)

    @torch.no_grad()
    def ei(self, y_best):
        return self.criterion.ei(self.logits, y_best)

    @torch.no_grad()
    def pi(self, y_best):
        return self.criterion.pi(self.logits, y_best)