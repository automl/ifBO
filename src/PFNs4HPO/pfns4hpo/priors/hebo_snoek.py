import time
import functools
import random
import math
import traceback
import warnings

import numpy as np
import torch
from torch import nn
import gpytorch
import botorch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior, UniformPrior, LogNormalPrior
from gpytorch.means import ZeroMean
from botorch.models.transforms.input import *
from gpytorch.constraints import GreaterThan

from botorch.models.transforms.input import Warp
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize

from . import utils
from ..utils import default_device, to_tensor
from .prior import Batch
from .utils import get_batch_to_dataloader
from pfns4hpo import encoders


def constraint_based_on_distribution_support(prior: torch.distributions.Distribution, device, sample_from_path):
    if sample_from_path:
        return None

    if hasattr(prior.support, 'upper_bound'):
        return gpytorch.constraints.Interval(to_tensor(prior.support.lower_bound,device=device),
                                             to_tensor(prior.support.upper_bound,device=device))
    else:
        return gpytorch.constraints.GreaterThan(to_tensor(prior.support.lower_bound,device=device))

def get_model(x, hyperparameters: dict):
    sample_from_path = None
    device = x.device
    num_features = x.shape[-1] - 1 # -1 because the last feature is the fidelity

    lengthscale_prior = UniformPrior(torch.tensor(0.0, device=device), torch.tensor(0.5, device=device))
    covar_module = gpytorch.kernels.MaternKernel(nu=3 / 2, ard_num_dims=num_features,
                                                active_dims=list(range(num_features)),
                                                lengthscale_prior=lengthscale_prior,
                                                lengthscale_constraint=\
                                                constraint_based_on_distribution_support(lengthscale_prior, device, sample_from_path))

    outputscale_prior = GammaPrior(concentration=hyperparameters.get('outputscale_concentration', .5), rate=hyperparameters.get('outputscale_rate', .5))
    covar_module = gpytorch.kernels.ScaleKernel(covar_module, outputscale_prior=outputscale_prior,
                                                outputscale_constraint=constraint_based_on_distribution_support(outputscale_prior, device, sample_from_path))

    if random.random() < float(hyperparameters.get('add_linear_kernel', True)):
        var_prior = UniformPrior(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        out_prior = UniformPrior(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        lincovar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(
                active_dims=list(range(num_features)),
                variance_prior=var_prior,
                variance_constraint=constraint_based_on_distribution_support(var_prior,device,sample_from_path),
        ),
            outputscale_prior=out_prior,
            outputscale_constraint=constraint_based_on_distribution_support(out_prior,device,sample_from_path),
        )
        covar_module = covar_module + lincovar_module
    
    warp_tf = Warp(
            indices=list(range(num_features)),
            concentration1_prior=LogNormalPrior(0.0, 0.75**5),
            concentration0_prior=LogNormalPrior(0.0, 0.75**5),
    )
    warp_tf.sample_from_prior("concentration1_prior")
    warp_tf.sample_from_prior("concentration0_prior")

    y = torch.zeros(x.shape[0], 1, device=device)
    model = SingleTaskMultiFidelityGP(x, y, iteration_fidelity=num_features, linear_truncated=False)
    model.mean_module = ZeroMean(x.shape[:-2])
    # model.covar_module.base_kernel.kernels[0] = covar_module 
    model.to(device)

    model = model.pyro_sample_from_prior().eval()
    # warp_tf = warp_tf.pyro_sample_from_prior().eval()
    return model, warp_tf

def sample_curves(idx_curves, epochs, nb_curves, num_features, unused_feature_likelihood, device, hyperparameters):

    local_x = torch.rand(nb_curves, num_features, device=device)
    configurations = local_x[idx_curves.long()-1]

    if unused_feature_likelihood:
        r = torch.rand(num_features)
        unused_feature_mask = r < unused_feature_likelihood
        if unused_feature_mask.all():
            unused_feature_mask[r.argmin()] = False
        used_local_x = configurations[...,~unused_feature_mask]
    else:
        used_local_x = configurations
    used_local_x = torch.cat([used_local_x, epochs.unsqueeze(-1)], dim=-1)

    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations',(True,True,True))):

        _get_model = lambda: get_model(used_local_x, hyperparameters)
        model, input_warp = _get_model()

        successful_sample = 0
        throwaway_share = 0.
        while successful_sample < 1:
            with gpytorch.settings.prior_mode(True):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wrap_config = input_warp(used_local_x)
                        d = model(wrap_config)
                        sample = d.sample()
                except Exception as e:
                    successful_sample -= 1
                    model, input_warp = _get_model()
                    if successful_sample < -100:
                        print(f'Could not sample from model {successful_sample} after {successful_sample} attempts. {e}')
                        raise e
                    continue

                target_values = sample
                successful_sample = True

            return configurations, target_values


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters=None,
              single_eval_pos=None,
              verbose=False, **kwargs):
    '''
    This function is very similar to the equivalent in .fast_gp. The only difference is that this function operates over
    a mixture of GP priors.
    :param batch_size:
    :param seq_len:
    :param num_features:
    :param device:
    :param hyperparameters:
    :param for_regression:
    :return:
    '''
    assert single_eval_pos <= seq_len
    assert num_features >= 2
    EPS = 10**-9
    verbose = hyperparameters.get("verbose", False)

    num_params = np.random.randint(1, num_features - 2)

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

        ok = False
        unused_feature_likelihood = np.random.rand()
        while not ok:
            configurations, target_values = sample_curves(idx_curves, epochs_per_curve, nb_curves, num_params, unused_feature_likelihood, device, hyperparameters)
            ok = torch.all(torch.isfinite(target_values))
        if verbose:
            print("Time taken for model and curve generation: ", time.time() - start_time)

        start_time = time.time()
        x[:, idx_batch, 0] = idx_curves
        x[:, idx_batch, 1] = epochs_per_curve
        x[:, idx_batch, 2:] = configurations
        y[:, idx_batch] = target_values
        x[:, idx_batch, 1] = x[:, idx_batch, 1] / max_epochs

        x[single_eval_pos:, idx_batch, 0] = torch.where(
            torch.isin(x[single_eval_pos:, idx_batch, 0], x[:single_eval_pos, idx_batch, 0]), 
            x[single_eval_pos:, idx_batch, 0], 
            torch.zeros_like(x[single_eval_pos:, idx_batch, 0])
        )
        if verbose:
            print("Time taken for final processing: ", time.time() - start_time)

    return Batch(x=x, y=y, target_y=y)

class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, num_features, emsize):
        super().__init__()
        self.epoch_enc = torch.nn.Sequential(
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            torch.nn.Linear(1, emsize, bias=False)
        )
        self.idcurve_enc = torch.nn.Embedding(1000 + 1, emsize)
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