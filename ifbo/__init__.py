import os
from dataclasses import dataclass
from typing import Optional

import torch

from .bar_distribution import BarDistribution
from .download import VERSION_MAP

@dataclass
class Curve:
    hyperparameters: torch.Tensor
    t: torch.Tensor
    y: Optional[torch.Tensor] = None

@dataclass(unsafe_hash=True)
class PredictionResult:
    logits: torch.Tensor
    criterion: BarDistribution

    @torch.no_grad()
    def likelihood(self, y_test):
        return -self.criterion(self.logits, y_test).squeeze(1)

    @torch.no_grad()
    def ucb(self):
        return self.criterion.ucb(self.logits, best_f=None).squeeze(1)

    @torch.no_grad()
    def ei(self, y_best):
        return self.criterion.ei(self.logits, f_best=y_best).squeeze(1)

    @torch.no_grad()
    def pi(self, y_best):
        return self.criterion.pi(self.logits, f_best=y_best).squeeze(1)
    
    @torch.no_grad()
    def quantile(self, q):
        return self.criterion.icdf(self.logits, q).squeeze(1)


from . import priors
from . import utils
from . import surrogate
from . import download