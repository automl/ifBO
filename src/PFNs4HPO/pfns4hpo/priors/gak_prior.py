import time
import functools
import random
import math
import traceback
import warnings
import random, math
import numpy as np
import torch
from torch import nn
import gpytorch
import botorch
from gpytorch.models import ExactGP
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior, UniformPrior, LogNormalPrior
from gpytorch.means import ZeroMean
from botorch.models.transforms.input import *
from gpytorch.constraints import GreaterThan
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from tslearn import metrics

import torch.nn.utils.rnn as rnn_utils

from . import utils
from ..utils import default_device, to_tensor
from .prior import Batch
from .utils import get_batch_to_dataloader
from pfns4hpo import encoders

def constraint_based_on_distribution_support(prior: torch.distributions.Distribution, device):

    if hasattr(prior.support, 'upper_bound'):
        return gpytorch.constraints.Interval(to_tensor(prior.support.lower_bound,device=device),
                                             to_tensor(prior.support.upper_bound,device=device))
    else:
        return gpytorch.constraints.GreaterThan(to_tensor(prior.support.lower_bound,device=device))
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
    elif type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

class DTWKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # apply lengthscale and remove 0 padding
        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale
        x1_ = [_[_!=0] if _.sum() else torch.FloatTensor([0.0]) for _ in x1.detach().cpu()]
        x2_ = [_[_!=0] if _.sum() else torch.FloatTensor([0.0]) for _ in x2.detach().cpu()]

        diff = metrics.cdist_gak(x1_, x2_)
        diff = torch.FloatTensor(diff).to(x1.device)
        return diff
    
class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, size_hp, size_lc):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.size_hp = size_hp
        self.size_lc = size_lc
        self.mean_module = ConstantMean()
        device = default_device
        
        lengthscale_prior_config = GammaPrior(torch.tensor(1., device=device), torch.tensor(1.5212245992840594, device=device))
        outputscale_prior = UniformPrior(0.05, 0.2)
        lengthscale_prior_budget = GammaPrior(torch.tensor(1., device=device), torch.tensor(1.5212245992840594, device=device))
        lengthscale_prior_lc = GammaPrior(torch.tensor(1., device=device), torch.tensor(1.5212245992840594, device=device))
        
        covar_config = gpytorch.kernels.RBFKernel(active_dims=torch.arange(size_hp), lengthscale_prior=lengthscale_prior_config, lengthscale_constraint=constraint_based_on_distribution_support(lengthscale_prior_config, device))
        
        covar_budget = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([size_hp]), lengthscale_prior=lengthscale_prior_budget, lengthscale_constraint=constraint_based_on_distribution_support(lengthscale_prior_budget, device))
        covar_lc = DTWKernel(active_dims=torch.arange(size_hp+1, size_hp+size_lc+1), lengthscale_prior=lengthscale_prior_lc, lengthscale_constraint=constraint_based_on_distribution_support(lengthscale_prior_lc, device))

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.AdditiveKernel(covar_config, covar_budget, covar_lc), outputscale_prior=outputscale_prior, outputscale_constraint=constraint_based_on_distribution_support(outputscale_prior, device))
        
        
        
    def forward(self, x, budgets, learning_curves):
        # pad learning curves with 0s to length size_lc
        learning_curves = torch.nn.functional.pad(learning_curves, (0, self.size_lc - learning_curves.size(1)))

        # Use the feature extractor to get the features
        data = torch.cat([x, budgets, learning_curves], axis=1)
        mean_x = self.mean_module(data)
        covar_x = self.covar_module(data)
        return MultivariateNormal(mean_x, covar_x)
    

def get_model(x, hyperparameters: dict, device=default_device):
    
    # likelihood definition
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    likelihood.register_prior("noise_prior",
                              LogNormalPrior(torch.tensor(hyperparameters.get('hebo_noise_logmean',-4.63), device=device),
                                             torch.tensor(hyperparameters.get('hebo_noise_std', 0.25), device=device)
                                             ),
                              "noise")

    model = GPModel(x, torch.rand(x.size(0)), likelihood, hyperparameters["num_features"], hyperparameters.get("size_lc", 50))
    
    model.to(device)
    likelihood.to(device)
    model = model.pyro_sample_from_prior()
    return model, model.likelihood

def generate_variable_length_sequences(budgets, device=default_device):
    """
    Generate a batch of variable-length sequences based on the provided budgets.

    Args:
    budgets (torch.Tensor): A 1D tensor of budget values.

    Returns:
    torch.Tensor: A padded tensor of sequences.
    """
    
    sequences = []
    p = -0.8
    for budget in budgets:
        if budget > 0:
            budget = random.choices(range(budget, 0, -1), [1 / math.pow(((budget - 0) - i), p) for i in range(budget - 0)])[0]
        
        if budget == 0:
            sequence = torch.zeros(size=(1,))
        else:
            sequence = torch.rand(size=(budget,))
        sequences.append(sequence)
    
    sequences.append(torch.empty(size=(50,)))  
    padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)[:-1]
    return padded_sequences.to(device)


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters={},
              batch_size_per_gp_sample=None, single_eval_pos=None,
              fix_to_range=None, equidistant_x=False, verbose=False, **kwargs):
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
    hyperparameters["num_features"] = num_features - 51
    fix_to_range = [1.5, -1.5]

    X, Y = [], []

    for _ in range(batch_size):

        with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations',(True,True,True))):

            budgets = torch.randint(1, 51, (seq_len,), device=device)
            budgets[torch.rand_like(budgets.float()) < 0.2] = 0
            # generate_budgets_with_total_sum(seq_len, np.random.uniform(), device=device)
            num_candidates = budgets.size(0)
            learning_curves = generate_variable_length_sequences(budgets, device=device)
            x = torch.rand(num_candidates, num_features, device=device)
            
            unused_feature_likelihood = hyperparameters.get('unused_feature_likelihood', True)
            
            if unused_feature_likelihood:
                r = torch.rand(x.size(1), device=device)
                unused_feature_mask = r < unused_feature_likelihood
                if unused_feature_mask.all():
                    unused_feature_mask[r.argmin()] = False
                configurations = x[...,~unused_feature_mask]
                hyperparameters["num_features"] = configurations.size(1)
            else:
                configurations = x
 
            get_model_and_likelihood = lambda: get_model(configurations, hyperparameters)
            model, likelihood = get_model_and_likelihood()

            if verbose: print(list(model.named_parameters()),
                            (list(model.input_transform.named_parameters()), model.input_transform.concentration1, model.input_transform.concentration0)
                                if model.input_transform is not None else None,
                            )

            successful_sample = 0
            while successful_sample < 1:
                with gpytorch.settings.prior_mode(True):
                    model.eval()
                    # print(x.device, device, f'{model.covar_module.base_kernel.lengthscale=}, {model.covar_module.base_kernel.lengthscale.device=}')
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            predictions = model(configurations, budgets.unsqueeze(-1), learning_curves)
                            predictions = likelihood(predictions).sample()

                    except (RuntimeError, ValueError) as e:
                        successful_sample -= 1
                        model, likelihood = get_model_and_likelihood()
                        if successful_sample < -100:
                            # print(f'Could not sample from model after {successful_sample} attempts. {e}')
                            raise e
                        continue

                    smaller_mask = predictions < fix_to_range[0]
                    larger_mask = predictions >= fix_to_range[1]
                    in_range_mask = ~ (smaller_mask & larger_mask).all(0)
                    
                    if in_range_mask.item():
                        successful_sample -= 1
                        model, likelihood = get_model_and_likelihood()
                        continue

                    successful_sample = True

                    feat = torch.cat([budgets.unsqueeze(-1), learning_curves, configurations], dim=-1)
                    # feat, targ = create_formatted_tensor(configurations, budgets, learning_curves, predictions, device=device)
                    # feat, targ = rearrange_tensor(feat, targ)
                    X.append(feat)
                    Y.append(predictions)
    
    x = torch.stack(X, dim=1).to(device).float()
    y = torch.stack(Y, dim=1).to(device).float()

    return Batch(x=x, y=y, target_y=y)

class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, num_features, emsize):
        super().__init__()
        self.budget_enc = torch.nn.Sequential(
            encoders.Normalize(1.0, 51.0),
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            torch.nn.Linear(1, emsize, bias=False)
        )
        self.lc_encoder = encoders.get_normalized_uniform_encoder(encoders.Linear)(50, emsize) # 50 is the number of epochs
        self.configuration_enc = encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear))(num_features - 51, emsize)

    def forward(self, x):
        # TODO
        # divided by the number of non zero values
        # to make the learning curve values in the same range as the other features.
        # lc = x[..., 1:51] * 50/lc[lc==0].size(-1)
        out = self.budget_enc(x[..., :1]) \
            + self.lc_encoder(x[..., 1:51]) \
            + self.configuration_enc(x[..., 51:])
        return out

def get_encoder():
    return lambda num_features, emsize: MultiCurvesEncoder(num_features, emsize)