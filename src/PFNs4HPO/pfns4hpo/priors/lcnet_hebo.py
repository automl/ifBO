import os
import torch
import math
import numpy as np
import random
import gpytorch
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
    res = torch.stack([y_a, y_b, y_c], dim=1)
    return res / torch.max(res.abs(), dim=2, keepdim=True).values

def get_lc_net_architecture(input_dimensionality: int, nepochs: int) -> torch.nn.Module:
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
            self.hebo = hebo_prior.get_model

        def forward(self, configurations, asymptotic):
            self.apply(init_weights)
            x = configurations
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            theta = torch.tanh(self.theta_layer(x))
            t = torch.arange(1, nepochs + 1)

            bf = bf_layer(theta, t)
            weights = torch.softmax(self.weight_layer(x), dim=1)

            # dirichlet = torch.distributions.dirichlet.Dirichlet(weights)
            # weights = dirichlet.sample()
            # asymptotic = torch.sigmoid(self.asymptotic_layer(x))
            residual = torch.tanh(torch.sum(bf * weights.unsqueeze(-1), dim=(1,), keepdim=False)) / 2

            """"
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                with gpytorch.settings.fast_computations((True, True, True)):
                    with gpytorch.settings.prior_mode(True):
                        xx = input.unsqueeze(0)
                        get_model_and_likelihood = lambda: hebo_prior.get_model(
                            xx,
                            torch.zeros(xx.shape[0], xx.shape[1]),
                            {},
                        )
                        model, likelihood = get_model_and_likelihood()
                        model.eval()
                        d = model(xx)
                        asymptotic = d.sample().transpose(0, 1)
                        asymptotic = torch.sigmoid(asymptotic)
            """

            mean = residual + torch.sigmoid(asymptotic)
            std = torch.sigmoid(self.sigma_layer(x))

            return mean + torch.normal(torch.zeros_like(mean), std / 100)

    return Architecture(n_inputs=input_dimensionality)


def get_curves_given_model(model, ncurves, nepochs, num_features, hyperparameters):
    hebo_data = hebo_prior.get_batch(1, seq_len=ncurves, num_features=num_features, hyperparameters=hyperparameters, device="cpu")
    curves = model(configurations=hebo_data.x.squeeze(1), asymptotic=hebo_data.target_y.squeeze(1))
    return hebo_data.x.squeeze(1), curves

# function producing batches for PFN training
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
    ncurves = hyperparameters.get("ncurves", 50)
    nepochs = hyperparameters.get("nepochs", 50)
    include_configurations = hyperparameters.get("include_configurations", True)

    # Random subsample of features
    num_features = num_features - 2

    # For now blindly use default Hebo params defined in PFNs4BO
    hyperparameters.update({
       'lengthscale_concentration': 1.2106559584074301,
       'lengthscale_rate': 1.5212245992840594,
       'outputscale_concentration': 0.8452312502679863,
       'outputscale_rate': 0.3993553245745406,
       'add_linear_kernel': False,
       'power_normalization': False,
       'hebo_warping': False,
       'unused_feature_likelihood': 0.3,
       'observation_noise': True
    })

    x = []
    y = []

    model = get_lc_net_architecture(input_dimensionality=num_features, nepochs=nepochs)
    alpha =  [min(
                math.ceil(_**(math.log(ncurves)
                              / math.log(ncurves * nepochs))),
                ncurves)
                for _ in range(1, ncurves * nepochs + 1)]
    epsilon = hyperparameters.get("epsilon", np.random.beta(0.5, 0.5))

    for i in range(batch_size):
        epoch = torch.zeros(nepochs * ncurves)
        id_curve = torch.zeros(nepochs * ncurves)
        curve_val = torch.zeros(nepochs * ncurves)
        idx = np.random.permutation(np.arange(1, ncurves + 1).repeat(nepochs))
        id_configurations = torch.zeros(nepochs * ncurves, num_features)

        # Get configurations and curves from HEBO and LCnet priors respectively
        configurations, curves = get_curves_given_model(model=model, ncurves=ncurves, nepochs=nepochs, num_features=num_features, hyperparameters=hyperparameters)

        assert len(curves) == len(configurations)

        nb_considered_curves = 0
        nb_indexes = torch.zeros(ncurves).type(torch.int64)
        for i in range(ncurves * nepochs):
            if alpha[i] > nb_considered_curves:
                selected = nb_considered_curves
                nb_considered_curves += 1
            elif np.random.uniform() < epsilon:
                candidates = torch.where(nb_indexes[:nb_considered_curves] < nepochs)[0]
                perm = torch.randperm(candidates.size(0))
                selected = candidates[perm[0]]
            else:
                candidates = torch.where(nb_indexes[:nb_considered_curves] < nepochs)[0]
                candidates_perf = curves[candidates, -1]
                selected = candidates[torch.argmax(candidates_perf)]

            id_curve[i] = selected
            curve_val[i] = curves[selected, nb_indexes[selected].long()]
            nb_indexes[selected] += 1
            epoch[i] = nb_indexes[selected]
            id_configurations[i] = configurations[selected]

        features = torch.stack([id_curve, epoch], dim=1)
        if include_configurations:
            features = torch.cat([features, id_configurations], dim=1)
        x.append(features)
        y.append(curve_val)

    x = torch.stack(x, dim=1).to(device).float()
    y = torch.stack(y, dim=1).to(device).float()

    return Batch(x=x, y=y, target_y=y)

class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, num_features, emsize):
        super().__init__()
        self.epoch_enc = torch.nn.Sequential(
            encoders.Normalize(1.0, 51.0),
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            torch.nn.Linear(1, emsize, bias=False)
        )
        self.idcurve_enc = torch.nn.Embedding(2500, emsize)
        self.configuration_enc = encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear))(num_features - 2, emsize)

    def forward(self, x):
        out = self.epoch_enc(x[..., 1:2]) \
            + self.idcurve_enc(x[..., :1].int()).squeeze(2) \
            + self.configuration_enc(x[..., 2:])
        return out

def get_encoder():
    return lambda num_features, emsize: MultiCurvesEncoder(num_features, emsize)

