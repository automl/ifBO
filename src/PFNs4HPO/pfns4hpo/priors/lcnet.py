import os
import torch
import math
import numpy as np
import random
import torch.nn as nn
from .utils import Batch
from . import hebo_prior
from pfns4hpo.utils import default_device
from pfns4hpo import encoders

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)

def vapor_pressure(x, a, b, c, *args):
    b_ = (b + 1) / 2 / 10
    a_ = (a + 1) / 2
    c_ = (c + 1) / 2 / 10
    return torch.exp(-a_ - b_ / (x + 1e-5) - c_ * torch.log(x)) - (torch.exp(a_ + b_))

def log_func(t, a, b, c, *args):
    a_ = (a + 1) / 2 * 5
    b_ = (b + 1) / 2
    c_ = (c + 1) / 2 * 10
    return (c_ + a_ * torch.log(b_ * t + 1e-10)) / 10.

def hill_3(x, a, b, c, *args):
    a_ = (a + 1) / 2
    b_ = (b + 1) / 2
    c_ = (c + 1) / 2 / 100
    return a_ * (1. / ((c_ / x + 1e-5) ** b_ + 1.))

def bf_layer(theta, t):
    y_a = vapor_pressure(t.unsqueeze(0), theta[:, 0].unsqueeze(-1), theta[:, 1].unsqueeze(-1), theta[:, 2].unsqueeze(-1))
    y_b = log_func(t.unsqueeze(0), theta[:, 3].unsqueeze(-1), theta[:, 4].unsqueeze(-1), theta[:, 5].unsqueeze(-1))
    y_c = hill_3(t.unsqueeze(0), theta[:, 6].unsqueeze(-1), theta[:, 7].unsqueeze(-1), theta[:, 8].unsqueeze(-1))
    return torch.stack([y_a, y_b, y_c], dim=1)

def get_lc_net_architecture(input_dimensionality: int, device: str = default_device) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=50):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, n_hidden)
            self.theta_layer = nn.Linear(n_hidden, 9)
            self.weight_layer = nn.Linear(n_hidden, 3)
            self.asymptotic_layer = nn.Linear(n_hidden, 1)
            self.sigma_layer = nn.Linear(n_hidden, 1)

            self.to(device)
            self.device = device

        def forward(self, input, nepochs):
            self.apply(init_weights)
            x = (input - 0.5) / math.sqrt(1/12)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            theta = torch.tanh(self.theta_layer(x))
            t = torch.arange(1, nepochs + 1).to(self.device)

            bf = bf_layer(theta, t)
            weights = torch.softmax(self.weight_layer(x), -1).unsqueeze(-1)
            residual = torch.tanh(torch.sum(bf * weights, dim=(1,), keepdim=False)) / 2

            asymptotic = torch.sigmoid(self.asymptotic_layer(x))
            mean = residual + asymptotic
            sigma = torch.sigmoid(self.sigma_layer(x))
            sigma = (sigma - sigma.min() + 1e-5) / torch.randint(1, 100, (1,), device=self.device)[0]

            return mean + torch.normal(torch.zeros_like(mean), sigma)

    return Architecture(n_inputs=input_dimensionality)



def get_curves_given_model(model, ncurves, nepochs, num_features, hyperparameters, device=default_device):
    configurations = torch.rand(ncurves, num_features, device=device)
    # no grad 
    with torch.no_grad():
        curves = model(configurations, nepochs)
    return configurations, curves

@torch.no_grad()
def get_batch(
    batch_size,
    seq_len,
    num_features,
    single_eval_pos,
    device=default_device,
    hyperparameters=None,
    **kwargs,
):
    assert single_eval_pos <= seq_len
    assert num_features >= 2
    EPS = 10**-9
    verbose = hyperparameters.get("verbose", False)

    num_params = np.random.randint(1, num_features - 1) # beware upper bound is exclusive!

    x = torch.zeros(seq_len, batch_size, num_params + 2, device=device)
    y = torch.zeros(seq_len, batch_size, device=device)

    ids = torch.arange(seq_len, device=device)

    # Initialize containers for batch-dependent values
    n_levels_batch = torch.round(10 ** torch.rand(batch_size, device=device) * 3).type(torch.int)
    alpha_batch = 10 ** (torch.rand(batch_size, device=device) * -3 - 1)
    weights_batch = torch.empty((batch_size, seq_len), device=device)
    for i, alpha in enumerate(alpha_batch):
        weights_batch[i] = torch.distributions.Gamma(alpha, alpha).sample((seq_len,))
    weights_batch += EPS

    # Normalize weights for each batch
    p_batch = weights_batch / torch.sum(weights_batch, dim=1, keepdim=True)

    model = get_lc_net_architecture(input_dimensionality=num_params, device=device)
    import time

    for idx_batch in range(batch_size):
        if verbose:
            print("*" * 50)
        start_time = time.time()
        n_levels = n_levels_batch[idx_batch].item()
        p = p_batch[idx_batch, :]

        all_levels = ids.repeat(n_levels)
        all_p = p.repeat(n_levels) / n_levels

        rand = torch.empty_like(all_p, device=device).uniform_()
        idx = (rand.log()/all_p).topk(k=seq_len).indices
        ordering = all_levels[idx]
        if verbose:
            print("Time taken for sampling: ", time.time() - start_time)

        idx_curves = torch.zeros(seq_len, device=device, dtype=torch.int)
        map_cid = {}
        for i, cid in enumerate(ordering):
            if cid.item() not in map_cid:
                map_cid[cid.item()] = len(map_cid) + 1
            idx_curves[i] = map_cid[cid.item()]

        if verbose:
            print("Time taken for remapping: ", time.time() - start_time)

        start_time = time.time()
        epochs_per_curve = torch.zeros(seq_len, device=device, dtype=torch.long)
        nepoch_all = torch.bincount(idx_curves)
        cutoff_all = torch.bincount(idx_curves, weights=(ids < single_eval_pos).float())
        for uid in range(1, idx_curves.max() + 1):
            mask = (idx_curves == uid)
            nepoch = nepoch_all[uid].item()
            cutoff = cutoff_all[uid].int().item()
            if cutoff == nepoch:
                epochs_per_curve[mask] = torch.arange(1, cutoff + 1, device=device)
            elif cutoff < nepoch:
                epochs_per_curve[mask] = torch.where(ids[mask] < single_eval_pos,
                    torch.arange(1, nepoch + 1, device=device),
                    torch.randint(cutoff + 1, n_levels + 1, (nepoch,), device=device)
                )
            else:
                raise ValueError("cutoff > nepoch")
        if verbose:
            print("Time taken for epoch per curve calculation: ", time.time() - start_time)

        start_time = time.time()
        nb_curves = idx_curves.max().item()
        max_epochs = epochs_per_curve.max()

        configurations, curves = get_curves_given_model(model=model, 
                                                        ncurves=nb_curves, 
                                                        nepochs=max_epochs, 
                                                        num_features=num_params, 
                                                        hyperparameters=hyperparameters)  
        if verbose:
            print("Time taken for model and curve generation: ", time.time() - start_time)

        start_time = time.time()
        x[:, idx_batch, 0] = idx_curves
        x[:, idx_batch, 1] = epochs_per_curve
        x[:, idx_batch, 2:] = configurations[x[:, idx_batch, 0].long()-1]
        y[:, idx_batch] = curves[x[:, idx_batch, 0].long()-1, x[:, idx_batch, 1].long()-1]
        x[:, idx_batch, 1] = x[:, idx_batch, 1] / max_epochs

        x[single_eval_pos:, idx_batch, 0] = torch.where(
            torch.isin(x[single_eval_pos:, idx_batch, 0], x[:single_eval_pos, idx_batch, 0]), 
            x[single_eval_pos:, idx_batch, 0], 
            torch.zeros_like(x[single_eval_pos:, idx_batch, 0])
        )
        if verbose:
            print("Time taken for final processing: ", time.time() - start_time)

    return Batch(x=x.float(), y=y.float(), target_y=y.float())


class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, num_features, emsize):
        super().__init__()
        self.epoch_enc = torch.nn.Sequential(
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            torch.nn.Linear(1, emsize, bias=False)
        )
        self.idcurve_enc = torch.nn.Embedding(1000 + 1, emsize)
        norms = torch.norm(self.idcurve_enc.weight, p=2, dim=1).detach()
        self.idcurve_enc.weight.data.div_(norms.unsqueeze(1))
        self.configuration_enc = encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear))(num_features - 2, emsize)
        

    def forward(self, x):
        out = self.idcurve_enc(x[..., :1].int()).squeeze(2)
        out = out + self.epoch_enc(x[..., 1:2]) 
        try:
            out = out + self.configuration_enc(x[..., 2:])
        except Exception as e:
            print(x.shape, self.epoch_enc, self.idcurve_enc, self.configuration_enc, out.shape)
            raise e
        return out

def get_encoder():
    return lambda num_features, emsize: MultiCurvesEncoder(num_features, emsize)

