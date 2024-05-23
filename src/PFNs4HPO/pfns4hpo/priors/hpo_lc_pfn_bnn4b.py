"""
CHANGELOG: Increase the diversity in curve relationships
- remove Kaiming initialization
- add random scaling of the inputs + bias (length-scale and parameter importance)
- add the 1'st derivative of gaussian as activation to encourage the highest/lowest values to occur inside of extrema
- add random scaling of the outputs + bias (curve property sensitivity & offset)
"""

import os
import torch
import math
import numpy as np
import random
from scipy.stats import norm, beta, gamma, expon
from pfns4hpo.encoders import Normalize
from pfns4hpo.priors.utils import Batch
from pfns4hpo.priors import hebo_prior
from pfns4hpo.utils import default_device
from pfns4hpo import encoders

def pow3(x, a, c, alpha, *args):
    return c - a * np.power(50*x+1, -alpha)


class GaussianDeriv(torch.nn.Module): 
    def __init__(self): 
        super(GaussianDeriv, self).__init__() 
  
    def forward(self, x): 
        return -x * ((-x**2+1)/2).exp()


class DatasetPrior:

    output_sorted = None

    def _get_model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.num_inputs+1, self.num_hidden, bias=False),
            GaussianDeriv(),
            torch.nn.Linear(self.num_hidden, self.num_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.num_hidden, self.num_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.num_hidden, self.num_outputs, bias=False)
        )

    def _output_for(self, input):
        with torch.no_grad():
            # normalize the inputs
            input = self.normalizer(input)
            # reweight the inputs for parameter importance
            input = input*self.input_weights
            # apply the model produce the output
            output = self.model(input.float())
            # rescale and shift outputs to account for parameter sensitivity
            #output = output * self.output_sensitivity + self.output_offset 
            return output
        

    def __init__(self, num_params, num_outputs):
        self.num_features = num_params
        self.num_outputs = num_outputs
        self.num_inputs = num_params+1
        self.num_hidden = 50

        N_datasets = 10000
        N_per_dataset = 1

        self.normalizer = Normalize(0.5, math.sqrt(1 / 12))

        if DatasetPrior.output_sorted is None:
            # generate 1M samples to approximate the CDF of the BNN output distribution
            output = torch.zeros((N_datasets, N_per_dataset, num_outputs))
            input = torch.from_numpy(np.random.uniform(size=(N_datasets, N_per_dataset, self.num_inputs))).to(torch.float32)
            input = torch.cat((input, torch.ones((N_datasets, N_per_dataset, 1))), -1)  # add ones as input bias 
            with torch.no_grad():
                for i in range(N_datasets):
                    if i % 100 == 99:
                        print(f"{i+1}/{N_datasets}")
                    # sample a new dataset
                    self.new_dataset()
                    for j in range(N_per_dataset):
                        output_ij = self._output_for(input[i,j])
                        output[i,j,:] = output_ij
                DatasetPrior.output_sorted = np.sort(torch.flatten(output).numpy())

        self.new_dataset()

    
    def new_dataset(self):
        # reinitialize all dataset specific random variables
        # reinit the parameters of the BNN
        self.model = self._get_model()
        # initial performance (after init)
        self.y0 = np.random.uniform()
        # the input weights (parameter importance & magnitude of aleatoric uncertainty on the curve) 
        param_importance = np.random.dirichlet([1]*(self.num_inputs-1) + [0.1]) # relative parameter importance
        lscale = np.exp(np.random.normal(2, 0.5)) # length scale ~ complexity of the landscape
        self.input_weights = np.concatenate((param_importance*lscale*self.num_inputs, np.full((1,),lscale)), axis=0)
        # the output weights (curve property sensitivity)
        self.output_sensitivity = np.random.uniform(size=(self.num_outputs,))
        self.output_offset = np.random.uniform((self.output_sensitivity-1)/2,(1-self.output_sensitivity)/2)

    def output_for_config(self, config, noise=True):
        # add aleatoric noise & bias
        input = np.concatenate((config, np.random.uniform(size=(*config.shape[:-1],1)) if noise else 0.5, np.ones((*config.shape[:-1],1))), -1)
        output = self._output_for(torch.from_numpy(input))
        return output.numpy()

    def uniform(self, bnn_output, a=0.0, b=1.0):
        indices = np.searchsorted(DatasetPrior.output_sorted, bnn_output, side='left')
        return (b-a) * indices / len(DatasetPrior.output_sorted) + a
    
    def normal(self, bnn_output, loc=0, scale=1):
        eps = 0.5 / len(DatasetPrior.output_sorted) # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1-eps)
        return norm.ppf(u, loc=loc, scale=scale)

    def beta(self, bnn_output, a=1, b=1, loc=0, scale=1):
        eps = 0.5 / len(DatasetPrior.output_sorted) # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1-eps)
        return beta.ppf(u, a=a, b=b, loc=loc, scale=scale)

    def gamma(self, bnn_output, a=1, loc=0, scale=1):
        eps = 0.5 / len(DatasetPrior.output_sorted) # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1-eps)
        return gamma.ppf(u, a=a, loc=loc, scale=scale)

    def exponential(self, bnn_output, scale=1):
        eps = 0.5 / len(DatasetPrior.output_sorted) # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1-eps)
        return expon.ppf(u, scale=scale)



def curve_prior(dataset, config):
    output = dataset.output_for_config(config)
    # pow3
    # sample the range c = (y_OPT - y0)
    c = dataset.uniform(output[0], a=dataset.y0, b=1.0)
    a = c - dataset.y0
    # sample the power rate alpha
    alpha = np.exp(dataset.normal(output[1], scale=2))
    # sample the measurement noise scale
    sigma = np.exp(dataset.normal(output[2], loc=-5, scale=1))

    def foo(x_):
        # x is a number from 0 to 1
        y_ = pow3(x_, a, c, alpha.item())
        noise = np.random.normal(size=x_.shape, scale=sigma)
        return y_ + noise
            
    return foo

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

    # assert num_features == 2
    assert num_features >= 2
    EPS = 10**-9
    
    num_params = np.random.randint(1, num_features - 1) # beware upper bound is exclusive!

    dataset_prior = DatasetPrior(num_params, 3)
    
    x = []
    y = []

    for i in range(batch_size):
        epoch = torch.zeros(seq_len)
        id_curve = torch.zeros(seq_len)
        curve_val = torch.zeros(seq_len)
        config = torch.zeros(seq_len, num_params)

        # determine the number of fidelity levels (ranging from 1: BB, up to seq_len)
        n_levels = int(np.round(10**np.random.uniform(0,3)))
        #print(f"n_levels: {n_levels}")

        # determine # observations/queries per curve
        # TODO: also make this a dirichlet thing
        alpha = 10**np.random.uniform(-4,-1)
        #print(f"alpha: {alpha}")
        weights = np.random.gamma(alpha,alpha,seq_len)+EPS
        p = weights / np.sum(weights)
        ids = np.arange(seq_len)
        all_levels = np.repeat(ids, n_levels)
        all_p = np.repeat(p, n_levels)/n_levels
        ordering = np.random.choice(all_levels, p=all_p, size=seq_len, replace=False)

        # calculate the cutoff/samples for each curve
        cutoff_per_curve = np.zeros((seq_len,), dtype=int)
        epochs_per_curve = np.zeros((seq_len,), dtype=int)
        for i in range(seq_len): # loop over every pos
            cid = ordering[i]
            epochs_per_curve[cid] += 1
            if i < single_eval_pos:
                cutoff_per_curve[cid] += 1

        # fix dataset specific random variables
        dataset_prior.new_dataset()

        # determine config, x, y for every curve
        curve_configs = []
        curve_xs = []
        curve_ys = []
        for cid in range(seq_len): # loop over every curve
            if epochs_per_curve[cid] > 0:
                # uniform random configurations
                c = np.random.uniform(size=(num_params,)) 
                curve_configs.append(c)
                # determine x (observations + query)
                x_ = np.zeros((epochs_per_curve[cid],))
                if cutoff_per_curve[cid] > 0: # observations (if any)
                    x_[:cutoff_per_curve[cid]] = np.arange(1,cutoff_per_curve[cid]+1)/n_levels
                if cutoff_per_curve[cid] < epochs_per_curve[cid]: # queries (if any)
                    x_[cutoff_per_curve[cid]:] = np.random.choice(np.arange(cutoff_per_curve[cid]+1, n_levels+1),
                                                                 size=epochs_per_curve[cid]-cutoff_per_curve[cid],
                                                                 replace=False)/n_levels
                curve_xs.append(x_)
                # sample curve
                curve = curve_prior(dataset_prior,c)
                # determine y's
                y_ = curve(torch.from_numpy(x_))
                curve_ys.append(y_)
            else:
                curve_configs.append(None)
                curve_xs.append(None)
                curve_ys.append(None)

        # construct the batch data element
        curve_counters = torch.zeros(seq_len).type(torch.int64)
        for i in range(seq_len):
            cid = ordering[i]
            if i < single_eval_pos or curve_counters[cid] > 0:
                id_curve[i] = cid + 1  # reserve ID 0 for queries
            else:
                id_curve[i] = 0  # queries for unseen curves always have ID 0
            epoch[i] = curve_xs[cid][curve_counters[cid]]
            config[i] = torch.from_numpy(curve_configs[cid])
            curve_val[i] = curve_ys[cid][curve_counters[cid]]
            curve_counters[cid] += 1
           
        x.append(torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1))
        y.append(curve_val)

    x = torch.stack(x, dim=1).to(device).float()
    y = torch.stack(y, dim=1).to(device).float()

    return Batch(x=x, y=y, target_y=y)


class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        seq_len = 1000
        self.normalizer = torch.nn.Sequential(
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
        )
        self.epoch_enc = torch.nn.Linear(1, out_dim, bias=False)
        self.idcurve_enc = torch.nn.Embedding(seq_len+1, out_dim)
        self.configuration_enc = encoders.get_variable_num_features_encoder(encoders.Linear)(in_dim-2, out_dim)

    def forward(self, *x, **kwargs):
        x = torch.cat(x, dim=-1)
        out = self.epoch_enc(self.normalizer(x[..., 1:2])) \
            + self.idcurve_enc(x[..., :1].int()).squeeze(2) \
            + self.configuration_enc(x[..., 2:])
        return out


def get_encoder():
    return lambda num_features, emsize: MultiCurvesEncoder(num_features, emsize)