import os
import torch
import math
import numpy as np
import random
import warnings
from scipy.stats import norm
from pfns4hpo.priors.utils import Batch
from pfns4hpo.utils import default_device
from pfns4hpo import encoders

import gpytorch
from gpytorch.priors.torch_priors import UniformPrior, LogNormalPrior
from scipy.stats import beta

class ExponentiallyDecayingKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize alpha and beta as learnable parameters
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.raw_noise = torch.nn.Parameter(torch.zeros(1))
        
        # Register priors for alpha and beta
        self.register_prior("alpha_prior", gpytorch.priors.LogNormalPrior(0, 1), "raw_alpha")
        self.register_prior("beta_prior", gpytorch.priors.LogNormalPrior(0, 1), "raw_beta")
        self.register_prior("noise_prior", gpytorch.priors.HorseshoePrior(0.001), "raw_noise")

    @property
    def alpha(self):
        return self.raw_alpha

    @property
    def beta(self):
        return self.raw_beta
    
    @property
    def noise(self):
        # Apply a softplus transformation to ensure positivity
        return self.raw_noise

    def forward(self, x1, x2, diag=False, **params):
        dist = x2.unsqueeze(1) + x1.unsqueeze(-1)
        dist = dist.squeeze(-1)
        decay_term = (self.beta**self.alpha) / (dist + self.beta)**self.alpha
        # Additive white noise kernel
        noise_term = (x2.unsqueeze(1) == x1.unsqueeze(-1)).int().squeeze(-1) * self.noise

        return decay_term + noise_term


class IcdfBetaScaler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x, a, b)
        return torch.tensor(beta.ppf(x, a.item(), b.item()))

    @staticmethod
    def backward(ctx, grad_output):
        x, a, b = ctx.saved_tensors
        x = x
        a = a.item()
        b = b.item()
        if x < h:
            grad_x = grad_output * torch.tensor((beta.ppf(x+h, a, b) - beta.ppf(x, a, b)) / h)
        elif x + h > 1.0:
            grad_x = grad_output * torch.tensor((beta.ppf(x, a, b) - beta.ppf(x-h, a, b)) / h)
        else:
            grad_x = grad_output * torch.tensor((beta.ppf(x+h, a, b) - beta.ppf(x-h, a, b)) / (2.0 * h))

        if a < h:
            grad_a = grad_output * torch.tensor((beta.ppf(x, a+h, b) - beta.ppf(x, a, b)) / h)
        else:
            grad_a = grad_output * torch.tensor((beta.ppf(x, a+h, b) - beta.ppf(x, a-h, b)) / (2.0 * h))

        if b < h:
            grad_b = grad_output * torch.tensor((beta.ppf(x, a, b+h) - beta.ppf(x, a, b)) / h)
        else:
            grad_b = grad_output * torch.tensor((beta.ppf(x, a, b+h) - beta.ppf(x, a, b-h)) / (2.0 * h))

        return grad_x, grad_a, grad_b


class FreezeThaw(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(FreezeThaw, self).__init__(train_x, train_y, likelihood)
        
        # configurations
        self.mean_module = gpytorch.means.ConstantMean(constant_prior=UniformPrior(0., 1.))
        self.covar_module_cfg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=5 / 2, lengthscale_prior=LogNormalPrior(0, 1)), output_scale=LogNormalPrior(0, 1))
        
        # learning cuves
        self.covar_module_lc = ExponentiallyDecayingKernel()
        
        # input wrapping
        self.register_parameter("beta_a", torch.nn.Parameter(torch.tensor(0.5)))
        self.register_parameter("beta_b", torch.nn.Parameter(torch.tensor(0.5)))
        self.register_prior("beta_a_prior", LogNormalPrior(0, 1), "beta_a")
        self.register_prior("beta_b_prior", LogNormalPrior(0, 1), "beta_b")

    def forward(self, configurations, epochs):
        scaled_configurations = IcdfBetaScaler.apply(configurations, self.beta_a, self.beta_b).type(torch.FloatTensor)

        mean_cfg = self.mean_module(scaled_configurations)
        covar_cfg = self.covar_module_cfg(scaled_configurations)
        
        mean_lc = gpytorch.distributions.MultivariateNormal(mean_cfg, covar_cfg).sample()
        covar_lc = self.covar_module_lc(epochs)

        return gpytorch.distributions.MultivariateNormal(mean_lc.repeat(1, covar_lc.shape[-1]), covar_lc)


def sample_curves(nb_configs = 50, config_dim=10, max_epochs = 50):
    # sample a collection of curves
    curves_configurations = torch.rand(nb_configs, 1, config_dim)
    epochs = torch.linspace(1, max_epochs+1, 50).unsqueeze(0).unsqueeze(-1).repeat(nb_configs, 1, 1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = FreezeThaw(None, None, likelihood)
    model = model.pyro_sample_from_prior()
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curves = model(curves_configurations, epochs).sample()

    return curves_configurations.squeeze(1), torch.sigmoid(curves)


import time

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

    assert num_features >= 2

    nepochs = 50
    ncurves = random.randint(int(seq_len / nepochs), seq_len)
    num_params = np.random.randint(1, num_features - 2)

    x = torch.zeros(seq_len, batch_size, num_params + 2, device=device)
    y = torch.zeros(seq_len, batch_size, device=device)

    for idx in range(batch_size):
        start_time = time.time()
        epoch = torch.zeros(seq_len)
        id_curve = torch.zeros(seq_len)
        curve_val = torch.zeros(seq_len)
        config = torch.zeros(seq_len, num_params)

        # sample a collection of curves
        ok = False
        while not ok:
            curve_configs, curves = sample_curves(nb_configs = ncurves, max_epochs = nepochs, config_dim=num_params)
            ok = torch.all(torch.isfinite(curves))
        # print(f"Sampling curves took {time.time() - start_time} seconds")

        start_time = time.time()
        ordering = hyperparameters.get("ordering", random.choice(["URS", "BFS", "DFS", "SoftGreedy"]))
        ids = np.arange(ncurves)
        np.random.shuffle(ids)  # randomize the order of IDs to avoid learning weird biases
        cutoff = torch.zeros(ncurves).type(torch.int64)
        # print(f"Ordering and setup took {time.time() - start_time} seconds")

        start_time = time.time()
    
        if ordering == "URS":
            candidates = np.arange(ncurves).repeat(nepochs)
            np.random.shuffle(candidates)
            selected_configs = candidates[:seq_len]
        elif ordering == "BFS":
            selected_configs = np.arange(seq_len) % ncurves
        elif ordering == "DFS":
            selected_configs = np.arange(seq_len) // nepochs
        elif ordering == "SoftGreedy":
            selected_configs = np.zeros(seq_len, dtype=np.int64)
            _cutoff = np.zeros(ncurves, dtype=np.int64)
            u = np.random.uniform(size=seq_len)
            p_new = 10**np.random.uniform(-3, 0, size=seq_len)
            greediness = 10**np.random.uniform(-4, 4)  # Select prob 'best' / 'worst' possible (1 = URS)

            for i in range(seq_len):
                new_candidates = np.where(_cutoff == 0)[0]
                candidates = np.where((_cutoff < nepochs) & (_cutoff > 0))[0]

                if u[i] < p_new[i] and len(new_candidates) > 0:
                    selected_configs[i] = np.random.choice(new_candidates)
                else:
                    if len(candidates) == 0:
                        selected_configs[i] = np.random.choice(new_candidates)
                    else:
                        selected_cutoff = np.random.randint(nepochs)
                        values = curves[candidates, np.minimum(selected_cutoff, _cutoff[candidates]-1)]
                        sm_values = np.power(greediness, values)
                        selected_configs[i] = random.choices(candidates, weights=sm_values)[0]
                _cutoff[selected_configs[i]] += 1

        for i, selected in enumerate(selected_configs):
            id_curve[i] = ids[selected]
            curve_val[i] = curves[selected][cutoff[selected]]
            config[i] = curve_configs[selected]
            cutoff[selected] += 1
            epoch[i] = cutoff[selected]
        # print(f"Selection took {time.time() - start_time} seconds", ordering)

        if hyperparameters.get("transform_unseen_curves", True):
            start_time = time.time()
            # unseen curve to id=0
            id_curve += 1
            id_curve[single_eval_pos:] = torch.where(
            torch.isin(id_curve[single_eval_pos:], id_curve[:single_eval_pos]), 
            id_curve[single_eval_pos:], 
            torch.zeros_like(id_curve[single_eval_pos:])
            )
            # print(f"Unseen curve processing took {time.time() - start_time} seconds")

        x[:, idx, 0] = id_curve.to(device)
        x[:, idx, 1] = epoch.to(device) / nepochs
        x[:, idx, 2:] = config.to(device)
        y[:, idx] = curve_val.to(device)

        # print("*" * 80)

    return Batch(x=x, y=y, target_y=y)


class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.normalizer = torch.nn.Sequential(
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
        )
        self.epoch_enc = torch.nn.Linear(1, out_dim, bias=False)
        self.idcurve_enc = torch.nn.Embedding(1001, out_dim)
        self.configuration_enc = encoders.get_variable_num_features_encoder(encoders.Linear)(in_dim-2, out_dim)

    def forward(self, *x, **kwargs):
        try:
            x = torch.cat(x, dim=-1)
            out = self.idcurve_enc(x[..., :1].int()).squeeze(2) 
            out += self.epoch_enc(self.normalizer(x[..., 1:2])) 
            out += self.configuration_enc(x[..., 2:])
            return out
        except:
            import pudb; pudb.set_trace()

def get_encoder():
    return lambda num_features, emsize: MultiCurvesEncoder(num_features, emsize)