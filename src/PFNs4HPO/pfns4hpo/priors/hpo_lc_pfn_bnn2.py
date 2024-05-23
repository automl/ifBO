import os
import torch
import math
import numpy as np
import random
from scipy.stats import norm
from pfns4hpo.priors.utils import Batch
from pfns4hpo.priors import hebo_prior
from pfns4hpo.utils import default_device
from pfns4hpo import encoders

def pow3(x, a, c, alpha, *args):
    return c - a * torch.pow(50*x+1, -alpha)


class DatasetPrior:

    output_sorted = None

    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def __init__(self, num_features, num_outputs):
        self.num_features = num_features
        self.num_outputs = num_outputs
        
        self.num_inputs = 2*(num_features+2)
        num_hidden = 100
        N = 1000
        
        self.model = torch.nn.Sequential(
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            torch.nn.Linear(self.num_inputs, num_hidden),
            torch.nn.ELU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ELU(),
            torch.nn.Linear(num_hidden, self.num_outputs)
        )

        if DatasetPrior.output_sorted is None:
            # generate samples to approximate the CDF of the BNN output distribution
            output = torch.zeros((N, num_outputs))
            input = torch.from_numpy(np.random.uniform(size=(N,self.num_inputs))).to(torch.float32)
            with torch.no_grad():
                for i in range(N):
                    self.model.apply(DatasetPrior.init_weights)
                    output[i] = self.model(input[i])
            
            DatasetPrior.output_sorted = np.sort(torch.flatten(output).numpy())

        # fix the parameters of the BNN
        self.model.apply(DatasetPrior.init_weights)

        # fix other dataset specific
        self.input_features = np.random.uniform(size=(self.num_inputs,))
        p_alloc = np.random.dirichlet(tuple([1 for _ in range(self.num_features)] + [1, 1]))
        self.alloc = np.random.choice(self.num_features+2, size=(self.num_inputs,), p=p_alloc)
        #print(self.alloc)

    
    def input_for_config(self, config):
        input_noise = np.random.uniform(size=(self.num_inputs,))
        input = torch.zeros((self.num_inputs,))
        for j in range(self.num_inputs):
            if self.alloc[j] < self.num_features:
                input[j] = config[self.alloc[j]]
            elif self.alloc[j] == self.num_features:
                input[j] = self.input_features[j]
            else:
                input[j] = input_noise[j]
        return input
        
    def output_for_config(self, config):
        input = self.input_for_config(config)
        return self.model(input)

    def uniform(self, bnn_output, a=0.0, b=1.0):
        indices = np.searchsorted(DatasetPrior.output_sorted, bnn_output, side='left')
        return (b-a) * indices / len(DatasetPrior.output_sorted) + a
    
    def normal(self, bnn_output, loc=0, scale=1):
        eps = 0.5 / len(DatasetPrior.output_sorted) # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1-eps)
        return norm.ppf(u, loc=loc, scale=scale)


def curve_prior(c_minus_a, dataset, config):
    output = dataset.output_for_config(config).numpy()
    c = dataset.uniform(output[0], a=c_minus_a, b=1.0)
    a = c - c_minus_a
    alpha = np.exp(dataset.normal(output[1], scale=2))
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
    
    num_params = np.random.randint(1, num_features - 1) # beware upper bound is exclusive!
    
    x = []
    y = []

    if hyperparameters.get("return_curves",False):
        batch_curves = []

    for i in range(batch_size):
        epoch = torch.zeros(seq_len)
        id_curve = torch.zeros(seq_len)
        curve_val = torch.zeros(seq_len)
        config = torch.zeros(seq_len, num_params)

        # determine # samples/queries per curve
        alpha = 10**np.random.uniform(-3,1)
        p = np.random.dirichlet([alpha]*seq_len)
        epochs_per_curve = np.random.multinomial(seq_len, p)

        # sample a collection of curves
        dataset = DatasetPrior(num_params, 3)
        curve_configs = []
        curve_xs = []
        curve_ys = []
        if hyperparameters.get("return_curves",False):
            curves = []
        
        # epoch 0 performance (performance of a randomly initialized model, before training)
        c_minus_a = np.random.uniform() 
        
        for i in range(seq_len):
            if epochs_per_curve[i] > 0:
                # uniform random configurations
                c = np.random.uniform(size=(num_params,)) 
                curve_configs.append(c)
                # uniform random x
                x_ = np.sort(np.random.uniform(size=(int(epochs_per_curve[i]),)))
                curve_xs.append(x_)
                # y
                curve = curve_prior(c_minus_a,dataset,c)
                curve_ys.append(curve(torch.from_numpy(x_)))
                if hyperparameters.get("return_curves",False):
                    curves.append(curve)
            else:
                curve_configs.append(None)
                curve_xs.append([])
                curve_ys.append([])
                if hyperparameters.get("return_curves",False):
                    curves.append(None)

        if hyperparameters.get("return_curves",False):
            batch_curves.append(curves)
        
        ids = np.arange(seq_len)
        # Repeat each element according to its count
        ordering = np.repeat(ids, epochs_per_curve)
        # Shuffle the array
        np.random.shuffle(ordering)

        # construct the batch data
        cutoff = torch.zeros(seq_len).type(torch.int64)
        for i in range(seq_len):
            cid = ordering[i]
            id_curve[i] = cid
            epoch[i] = curve_xs[cid][cutoff[cid]]
            config[i] = torch.from_numpy(curve_configs[cid])
            curve_val[i] = curve_ys[cid][cutoff[cid]]
            cutoff[cid] += 1 
           
        x.append(torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1))
        y.append(curve_val)

    x = torch.stack(x, dim=1).to(device).float()
    y = torch.stack(y, dim=1).to(device).float()

    if hyperparameters.get("return_curves",False):
        return Batch(x=x, y=y, target_y=y), batch_curves
    else:
        return Batch(x=x, y=y, target_y=y)


class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        seq_len = 1000
        self.normalizer = torch.nn.Sequential(
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
        )
        self.epoch_enc = torch.nn.Linear(1, out_dim, bias=False)
        self.idcurve_enc = torch.nn.Embedding(seq_len, out_dim)
        self.configuration_enc = encoders.get_variable_num_features_encoder(encoders.Linear)(in_dim-2, out_dim)

    def forward(self, *x, **kwargs):
        x = torch.cat(x, dim=-1)
        out = self.epoch_enc(self.normalizer(x[..., :1])) \
            + self.idcurve_enc(x[..., 1:2].int()).squeeze(2) \
            + self.configuration_enc(x[..., 2:])
        return out


def get_encoder():
    return lambda num_features, emsize: MultiCurvesEncoder(num_features, emsize)